![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-orange?logo=huggingface&logoColor=white)
![Status](https://img.shields.io/badge/status-in%20development-yellow)
![CI](https://github.com/LukeSantossz/tweet-sentiment-analysis/actions/workflows/ci.yml/badge.svg)

# tweet-sentiment-analysis

> Fine-tuned RoBERTa model for 3-class sentiment classification on tweets, evaluated against the TweetEval benchmark.

## Why This Exists

Generic sentiment models underperform on social media text. Tweets contain abbreviations, slang, @mentions, hashtags, and emojis that break assumptions built into models trained on formal corpora. This project fine-tunes a Twitter-specialized RoBERTa variant (`cardiffnlp/twitter-roberta-base-sentiment`) on the TweetEval benchmark dataset to classify tweets as **negative**, **neutral**, or **positive** — and measures the gain over the zero-shot baseline.

The zero-shot baseline achieves 70% accuracy and 0.71 macro F1. The fine-tuning pipeline is built to surpass these numbers on the same test split (12,284 samples).

## Architecture

```mermaid
flowchart LR
    subgraph Data
        A[TweetEval Dataset<br/>45.6K train · 2K val · 12.3K test]
    end

    subgraph Preprocessing
        B[src/preprocessing.py<br/>URLs → token · @mentions → @user<br/>emojis → text · hashtags · lowercase]
    end

    subgraph Training
        C[src/training.py<br/>HuggingFace Trainer API<br/>lr=2e-5 · 3 epochs · early stopping]
    end

    subgraph Model
        D[twitter-roberta-base-sentiment<br/>CardiffNLP · 125M params]
    end

    subgraph Evaluation
        E[Accuracy + Macro F1<br/>Confusion Matrix<br/>Per-class metrics]
    end

    A --> B --> C --> D --> E
```

## Engineering Decisions

| Decision | Rationale |
|----------|-----------|
| **Model: `twitter-roberta-base-sentiment`** | Pre-trained on ~58M tweets — domain-aligned, no need for domain adaptation from scratch. |
| **`max_length=128` tokens** | 99th percentile of token lengths is ~55. 128 is conservative but avoids any truncation artifacts. |
| **Macro F1 as primary metric** | Dataset is imbalanced (neutral ~45%, positive ~30%, negative ~22%). Macro F1 penalizes poor performance on minority classes. |
| **URLs replaced with `[URL]` token** | Preserves signal that a URL was present without introducing noise from the URL content itself. |
| **Emojis converted via `emoji.demojize()`** | Transforms emojis into descriptive text (e.g., `:fire:`) readable by the tokenizer, preserving sentiment signal. |
| **Early stopping with `patience=2`** | Prevents overfitting on the relatively small training set without manual epoch tuning. |
| **CPU-only PyTorch in CI** | Avoids ~2GB CUDA download in the pipeline. Slow tests requiring GPU/network are excluded via pytest marker. |
| **Ruff for linting** | Fast, Rust-based linter. Rules E/F/I with `line-length=120`. Notebooks excluded (not production code). |

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- (Optional) CUDA 11.x+ for GPU acceleration

### Installation

```bash
git clone https://github.com/LukeSantossz/tweet-sentiment-analysis.git
cd tweet-sentiment-analysis

python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### Running

```bash
# Run the fine-tuning script
python -m src.training

# Run tests
pytest tests/ -m "not slow" -v

# Run linter
ruff check . && ruff format --check .

# Launch Jupyter for analysis notebooks
jupyter notebook
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_DATASETS_CACHE` | Hugging Face Datasets cache directory | `~/.cache/huggingface/datasets` |
| `TRANSFORMERS_CACHE` | Pre-trained model cache directory | `~/.cache/huggingface/transformers` |
| `CUDA_VISIBLE_DEVICES` | GPU index(es) to use | `0` |

## Project Structure

```
tweet-sentiment-analysis/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Tweet cleaning pipeline (URLs, mentions, emojis, hashtags)
│   └── training.py                 # Fine-tuning script with HuggingFace Trainer API
├── tests/
│   ├── test_preprocessing.py       # 11 unit tests for preprocessing functions
│   └── test_training.py            # 10 tests for training module (config, metrics, constants)
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis: class distribution, text patterns
│   ├── 02_tokenization.ipynb       # Token length distribution, max_length validation
│   └── 03_inference_baseline.ipynb # Zero-shot baseline: 70% acc, 0.71 macro F1
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions: lint (ruff) + test (pytest)
├── .claude/                        # AI agent governance rules and project registry
├── pyproject.toml                  # Ruff and pytest configuration
├── requirements.txt                # Python dependencies
└── README.md
```

## Current Status

| Stage | Status | Details |
|-------|--------|---------|
| Exploratory Data Analysis | Done | Class imbalance identified, noise patterns mapped |
| Preprocessing Pipeline | Done | 6 cleaning functions, 11 passing tests |
| Tokenization Analysis | Done | max_length=128 validated at 99th percentile |
| Zero-shot Baseline | Done | 70% accuracy, 0.71 macro F1 |
| Training Script | Done | Trainer API, early stopping, CLI args |
| Training Module Tests | Done | 10 tests covering config, metrics, constants |
| CI Pipeline | Done | GitHub Actions with ruff + pytest |
| Fine-tuning Execution | Pending | Script ready, awaiting GPU execution |
| Comparative Evaluation | Pending | Baseline vs fine-tuned, per-class analysis |
| REST API (FastAPI) | Planned | POST /predict endpoint |
| Demo UI (Gradio) | Planned | Interactive frontend |
| Docker Containerization | Planned | Dockerfile + docker-compose |
