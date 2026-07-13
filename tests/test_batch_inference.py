"""Tests for the scale batch-inference script."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.batch_inference import (
    build_predictions,
    iter_text_chunks,
    parse_args,
    preprocess_chunk,
    run_batch_inference,
    throughput_per_second,
)
from src.training import DEFAULT_OUTPUT_DIR, LABEL_NAMES

_CHECKPOINT_PRESENT = Path(DEFAULT_OUTPUT_DIR).is_dir()


def _write_parquet(path, columns: dict):
    pq.write_table(pa.table(columns), path)


def test_parse_args_defaults():
    args = parse_args(["--input", "in.parquet", "--output", "out.parquet"])

    assert args.input == "in.parquet"
    assert args.output == "out.parquet"
    assert args.text_column == "text"
    assert args.batch_size == 64
    assert args.chunk_size == 10_000
    assert args.model == "./outputs/finetuned-model"


def test_parse_args_custom_values():
    args = parse_args(
        [
            "--input",
            "i",
            "--output",
            "o",
            "--text-column",
            "content",
            "--batch-size",
            "8",
            "--chunk-size",
            "500",
            "--model",
            "m",
        ]
    )

    assert args.text_column == "content"
    assert args.batch_size == 8
    assert args.chunk_size == 500
    assert args.model == "m"


def test_iter_text_chunks_streams_all_rows_in_order(tmp_path):
    path = tmp_path / "in.parquet"
    texts = ["a", "b", "c", "d", "e"]
    _write_parquet(path, {"text": texts})

    chunks = list(iter_text_chunks(str(path), "text", chunk_size=2))

    assert [len(c) for c in chunks] == [2, 2, 1]
    assert [row for chunk in chunks for row in chunk] == texts


def test_iter_text_chunks_coerces_null_to_empty(tmp_path):
    path = tmp_path / "in.parquet"
    _write_parquet(path, {"text": ["x", None, "y"]})

    rows = [row for chunk in iter_text_chunks(str(path), "text", chunk_size=10) for row in chunk]

    assert rows == ["x", "", "y"]


def test_iter_text_chunks_missing_column_raises(tmp_path):
    path = tmp_path / "in.parquet"
    _write_parquet(path, {"body": ["x"]})

    with pytest.raises(ValueError, match="text"):
        list(iter_text_chunks(str(path), "text", chunk_size=10))


def test_preprocess_chunk_applies_model_contract():
    # preprocess_for_model collapses @mentions/URLs but preserves case, hashtags, and emoji --
    # i.e. NOT the lowercasing/demojizing/[URL] bulk contract of clean_tweet_text.
    result = preprocess_chunk(["@joao http://x.co #Tag 😊 CAPS"])

    assert result == ["@user http #Tag 😊 CAPS"]


def test_build_predictions_maps_logits_to_names_and_scores():
    logits = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # argmax 0 -> sadness
            [0.0, 0.0, 0.0, 0.0, 0.0, 5.0],  # argmax 5 -> surprise
        ]
    )

    labels, scores = build_predictions(logits)

    assert labels == [LABEL_NAMES[0], LABEL_NAMES[5]]
    # softmax max of row 0: exp(2)/(exp(2)+5) = 0.5964...
    assert scores[0] == pytest.approx(np.exp(2) / (np.exp(2) + 5), rel=1e-4)
    assert 0.0 <= scores[1] <= 1.0
    assert scores[1] > scores[0]


def test_throughput_per_second():
    assert throughput_per_second(1000, 2.0) == pytest.approx(500.0)
    assert throughput_per_second(10, 0.0) == 0.0


def test_run_batch_inference_empty_input_writes_empty_output(tmp_path):
    # Zero-row input must still produce a valid, correctly-typed empty Parquet (no model load).
    in_path = tmp_path / "in.parquet"
    out_path = tmp_path / "out.parquet"
    _write_parquet(in_path, {"text": pa.array([], type=pa.string())})

    metrics = run_batch_inference(str(in_path), str(out_path))

    table = pq.read_table(out_path)
    assert table.num_rows == 0
    assert set(table.column_names) == {"text_original", "label", "score", "processing_time_ms"}
    assert metrics["rows"] == 0


def test_run_batch_inference_rejects_output_equal_to_input(tmp_path):
    path = tmp_path / "same.parquet"
    _write_parquet(path, {"text": ["a"]})

    with pytest.raises(ValueError, match="output"):
        run_batch_inference(str(path), str(path))


def test_run_batch_inference_empty_input_missing_column_raises(tmp_path):
    # The column check must run before the zero-row fast path, not be bypassed by it.
    in_path = tmp_path / "in.parquet"
    out_path = tmp_path / "out.parquet"
    _write_parquet(in_path, {"body": pa.array([], type=pa.string())})

    with pytest.raises(ValueError, match="text"):
        run_batch_inference(str(in_path), str(out_path))


@pytest.mark.slow
@pytest.mark.skipif(not _CHECKPOINT_PRESENT, reason="fine-tuned checkpoint (gitignored) not present")
def test_batch_inference_end_to_end(tmp_path):
    texts = ["i am so happy today", "this is terrible and sad", "@x check http://y.co 😊"]
    in_path = tmp_path / "in.parquet"
    out_path = tmp_path / "out.parquet"
    _write_parquet(in_path, {"text": texts})

    metrics = run_batch_inference(str(in_path), str(out_path), model_dir=DEFAULT_OUTPUT_DIR, batch_size=2, chunk_size=2)

    table = pq.read_table(out_path)
    assert table.num_rows == 3
    assert set(table.column_names) == {"text_original", "label", "score", "processing_time_ms"}
    assert table.column("text_original").to_pylist() == texts
    assert set(table.column("label").to_pylist()).issubset(set(LABEL_NAMES))
    assert all(0.0 <= s <= 1.0 for s in table.column("score").to_pylist())
    assert metrics["rows"] == 3
    assert metrics["tweets_per_second"] > 0
