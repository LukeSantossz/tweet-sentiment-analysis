"""
Fine-tuning script for emotion classification on dair-ai/emotion.

This module implements fine-tuning of the cardiffnlp/twitter-roberta-base model
on the dair-ai/emotion dataset using the Hugging Face Trainer API.

Hyperparameters:
    - learning_rate: 2e-5
    - num_train_epochs: 3
    - per_device_train_batch_size: 16
    - per_device_eval_batch_size: 32
    - warmup_steps: 500
    - weight_decay: 0.01
    - max_length: 128 tokens
    - evaluation_strategy: epoch
    - save_strategy: epoch
    - load_best_model_at_end: True
    - metric_for_best_model: f1_macro

Usage:
    python -m src.training
    python -m src.training --output_dir ./my_model --epochs 5
"""

import argparse

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .preprocessing import preprocess_for_model

MODEL_NAME = "cardiffnlp/twitter-roberta-base"
DATASET_NAME = "dair-ai/emotion"
DATASET_CONFIG = "split"
MAX_LENGTH = 128
DEFAULT_OUTPUT_DIR = "./outputs/finetuned-model"
SEED = 42

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def subset_size(available: int, requested: int | None) -> int:
    """Return how many rows to keep: all of them unless a smaller non-negative cap is requested."""
    if requested is None or requested < 0:
        return available
    return min(requested, available)


def load_tokenizer_and_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load the tokenizer and a sequence-classification model with a head sized to LABEL_NAMES.

    The backbone is a task-agnostic MLM (no pretrained classification head), so a randomly
    initialized head is added — the "newly initialized weights" warning is expected (ADR 0011).
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label={i: label for i, label in enumerate(LABEL_NAMES)},
        label2id={label: i for i, label in enumerate(LABEL_NAMES)},
    )
    return tokenizer, model


def load_emotion_dataset() -> DatasetDict:
    """Load the dair-ai/emotion dataset (config 'split': 16k/2k/2k, single-label, 6 classes)."""
    return load_dataset(DATASET_NAME, DATASET_CONFIG)


def tokenize_dataset(dataset: DatasetDict, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    """
    Tokenize the dataset using the provided tokenizer.

    Args:
        dataset: Raw dataset with 'text' column
        tokenizer: Hugging Face tokenizer

    Returns:
        Tokenized dataset ready for training
    """

    def tokenize_fn(examples):
        texts = [preprocess_for_model(text) for text in examples["text"]]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )
    return tokenized


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """
    Compute accuracy and macro F1-score for evaluation.

    Args:
        eval_pred: EvalPrediction with predictions and label_ids

    Returns:
        Dictionary with accuracy and f1_macro metrics
    """
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
    }


def create_training_args(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    fp16: bool = False,
    max_steps: int = -1,
    seed: int = SEED,
) -> TrainingArguments:
    """
    Create TrainingArguments with recommended hyperparameters.

    Args:
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate for AdamW optimizer
        per_device_train_batch_size: Training batch size per device
        per_device_eval_batch_size: Evaluation batch size per device
        warmup_steps: Number of warmup steps for learning rate scheduler
        weight_decay: Weight decay for regularization
        fp16: Enable fp16 mixed precision (GPU only)
        max_steps: Cap total optimizer steps (-1 = use epochs)
        seed: Random seed for reproducibility

    Returns:
        Configured TrainingArguments
    """
    return TrainingArguments(
        output_dir=output_dir,
        seed=seed,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_strategy="epoch",
        report_to="none",
        fp16=fp16,
        push_to_hub=False,
    )


def create_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> Trainer:
    """
    Create a Trainer instance with the given configuration.

    Args:
        model: Pre-trained model
        tokenizer: Tokenizer for the model
        training_args: Training configuration
        train_dataset: Tokenized training dataset
        eval_dataset: Tokenized validation dataset

    Returns:
        Configured Trainer instance
    """
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )


def train(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 16,
    per_device_eval_batch_size: int = 32,
    fp16: bool | None = None,
    max_steps: int = -1,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> dict[str, float]:
    """
    Execute the full fine-tuning pipeline.

    Args:
        output_dir: Directory to save model checkpoints
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        per_device_train_batch_size: Training batch size
        per_device_eval_batch_size: Evaluation batch size
        fp16: fp16 mixed precision; None auto-enables it when CUDA is available
        max_steps: Cap total optimizer steps (-1 = use epochs)
        max_train_samples: Use at most N training examples (smoke tests)
        max_eval_samples: Use at most N validation examples (smoke tests)

    Returns:
        Dictionary with final evaluation metrics
    """
    print(f"Loading model and tokenizer: {MODEL_NAME}")
    set_seed(SEED)
    tokenizer, model = load_tokenizer_and_model()

    print(f"Loading dataset: {DATASET_NAME}/{DATASET_CONFIG}")
    dataset = load_emotion_dataset()

    raw_train = dataset["train"]
    raw_eval = dataset["validation"]
    raw_train = raw_train.select(range(subset_size(len(raw_train), max_train_samples)))
    raw_eval = raw_eval.select(range(subset_size(len(raw_eval), max_eval_samples)))

    print(f"Tokenizing dataset with max_length={MAX_LENGTH}")
    splits = DatasetDict({"train": raw_train, "validation": raw_eval})
    tokenized = tokenize_dataset(splits, tokenizer)
    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]

    if fp16 is None:
        fp16 = torch.cuda.is_available()

    training_args = create_training_args(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=fp16,
        max_steps=max_steps,
    )

    trainer = create_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on validation set...")
    eval_results = trainer.evaluate()

    print(f"Saving best model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Validation F1 (macro): {eval_results['eval_f1_macro']:.4f}")
    print(f"Model saved to: {output_dir}")
    print("=" * 50)

    return eval_results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Twitter-RoBERTa for emotion classification on dair-ai/emotion"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Evaluation batch size per device",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable fp16 mixed precision (default: auto -- on when CUDA is available)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="Cap total optimizer steps (-1 = use epochs); useful for smoke tests",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Use at most N training examples (smoke tests)",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Use at most N validation examples (smoke tests)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        fp16=args.fp16,
        max_steps=args.max_steps,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
