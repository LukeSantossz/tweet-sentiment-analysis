"""Scale batch inference: GPU emotion predictions over a Parquet of tweets.

The model path normalizes the original text with ``preprocess_for_model`` (ADR 0009), not the
Rust bulk ``text_cleaned`` contract, so serving inputs match the training distribution. Input is
streamed in row chunks (pyarrow ``iter_batches``) and predictions are streamed out
(pyarrow ``ParquetWriter``) so the full dataset is never held in memory; the model is loaded once.
"""

import argparse
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .preprocessing import preprocess_for_model
from .training import DEFAULT_OUTPUT_DIR, LABEL_NAMES, MAX_LENGTH


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the batch-inference CLI arguments."""
    parser = argparse.ArgumentParser(description="Batch GPU inference over a Parquet of tweets.")
    parser.add_argument("--input", required=True, help="Input Parquet with an original-text column")
    parser.add_argument("--output", required=True, help="Output Parquet with predictions")
    parser.add_argument("--text-column", default="text", help="Name of the original-text column")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--chunk-size", type=int, default=10_000, help="Rows read/written per streamed chunk")
    parser.add_argument("--model", default=DEFAULT_OUTPUT_DIR, help="Path to the fine-tuned model directory")
    return parser.parse_args(argv)


def iter_text_chunks(input_path: str, text_column: str, chunk_size: int):
    """Yield lists of original-text strings, streamed from the Parquet in row chunks.

    Null values are coerced to empty strings so the output stays row-aligned with the input.
    """
    parquet_file = pq.ParquetFile(input_path)
    if text_column not in parquet_file.schema_arrow.names:
        raise ValueError(f"input Parquet has no '{text_column}' column; found {parquet_file.schema_arrow.names}")
    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=[text_column]):
        values = batch.column(0).to_pylist()
        yield ["" if value is None else str(value) for value in values]


def preprocess_chunk(texts: list[str]) -> list[str]:
    """Apply the model-input contract (``preprocess_for_model``) to a chunk of original text."""
    return [preprocess_for_model(text) for text in texts]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def build_predictions(logits: np.ndarray) -> tuple[list[str], list[float]]:
    """Map logits ``(n, num_labels)`` to emotion-name labels and max-softmax confidence scores."""
    probs = _softmax(np.asarray(logits, dtype=np.float64))
    indices = probs.argmax(axis=1)
    labels = [LABEL_NAMES[int(index)] for index in indices]
    scores = [float(score) for score in probs.max(axis=1)]
    return labels, scores


def throughput_per_second(rows: int, seconds: float) -> float:
    """Tweets processed per second; 0.0 when no time elapsed."""
    return rows / seconds if seconds > 0 else 0.0


def run_batch_inference(
    input_path: str,
    output_path: str,
    model_dir: str = DEFAULT_OUTPUT_DIR,
    text_column: str = "text",
    batch_size: int = 64,
    chunk_size: int = 10_000,
) -> dict:
    """Stream-infer emotions over ``input_path`` and write predictions to ``output_path``.

    Returns throughput metrics: number of rows, elapsed seconds, and tweets/second.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    total_rows = pq.ParquetFile(input_path).metadata.num_rows
    writer = None
    processed = 0
    start = time.perf_counter()

    with torch.no_grad(), tqdm(total=total_rows, desc="Inferring", unit="tweet") as progress:
        for chunk_texts in iter_text_chunks(input_path, text_column, chunk_size):
            preprocessed = preprocess_chunk(chunk_texts)
            labels: list[str] = []
            scores: list[float] = []
            times_ms: list[float] = []
            for batch_texts in DataLoader(preprocessed, batch_size=batch_size):
                batch_texts = list(batch_texts)
                batch_start = time.perf_counter()
                encoded = tokenizer(
                    batch_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
                ).to(device)
                logits = model(**encoded).logits.detach().cpu().numpy()
                elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
                batch_labels, batch_scores = build_predictions(logits)
                labels.extend(batch_labels)
                scores.extend(batch_scores)
                times_ms.extend([elapsed_ms / len(batch_texts)] * len(batch_texts))
                progress.update(len(batch_texts))

            table = pa.table(
                {
                    "text_original": chunk_texts,
                    "label": labels,
                    "score": scores,
                    "processing_time_ms": times_ms,
                }
            )
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            processed += len(chunk_texts)
            if device == "cuda":
                torch.cuda.empty_cache()

    if writer is not None:
        writer.close()

    total_time = time.perf_counter() - start
    tweets_per_second = throughput_per_second(processed, total_time)
    print(f"Processed {processed} tweets in {total_time:.2f}s ({tweets_per_second:,.0f} tweets/sec)")
    return {"rows": processed, "seconds": total_time, "tweets_per_second": tweets_per_second}


def main(argv=None) -> None:
    """CLI entry point."""
    args = parse_args(argv)
    run_batch_inference(
        args.input,
        args.output,
        model_dir=args.model,
        text_column=args.text_column,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
