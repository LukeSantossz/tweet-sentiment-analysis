![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Transformers-orange?logo=huggingface&logoColor=white)
![Status](https://img.shields.io/badge/status-in%20development-yellow)

# tweet-sentiment-analysis

> Sentiment classification pipeline for tweets using Hugging Face Transformers and the TweetEval benchmark.

## Overview

Tweets have distinct linguistic patterns compared to formal text — abbreviations, slang, mentions, hashtags, and emojis make sentiment analysis a non-trivial problem for generic NLP models. This project addresses that gap by fine-tuning a social-media-specialized transformer on the `cardiffnlp/tweet_eval` dataset (negative / neutral / positive) and evaluating it against the TweetEval benchmark baseline.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Dataset | `cardiffnlp/tweet_eval` (Hugging Face Datasets) |
| Model | BERT/RoBERTa family (`cardiffnlp/twitter-roberta-base-sentiment`) |
| Training | Hugging Face `Trainer` API |
| Preprocessing | Custom `src/preprocessing.py` + `emoji` library |
| Evaluation | `scikit-learn` (accuracy, macro F1) |
| Visualization | `matplotlib`, `seaborn` |
| Acceleration | PyTorch + CUDA (optional) |

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip
- (Optional) CUDA 11.x or higher for GPU acceleration

### Installation

```bash
# Clone the repository
git clone https://github.com/<username>/tweet-sentiment-analysis.git
cd tweet-sentiment-analysis

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running

```bash
# Start the Jupyter server
jupyter notebook

# Open notebooks in order:
# notebooks/01_eda.ipynb            - dataset exploratory analysis
# notebooks/02_tokenization.ipynb   - preprocessing and tokenization
```

**Environment variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_DATASETS_CACHE` | Hugging Face Datasets cache directory | `~/.cache/huggingface/datasets` |
| `TRANSFORMERS_CACHE` | Pre-trained model cache directory | `~/.cache/huggingface/transformers` |
| `CUDA_VISIBLE_DEVICES` | GPU index(es) to use | `0` |

## Project Structure

```
tweet-sentiment-analysis/
├── notebooks/
│   ├── 01_eda.ipynb              # EDA: class distribution across train/test/validation splits
│   └── 02_tokenization.ipynb     # Preprocessing and tokenization
├── src/
│   ├── __init__.py
│   └── preprocessing.py          # Text cleaning and normalization functions
├── tests/
│   └── test_preprocessing.py     # Unit tests for the preprocessing module
├── venv/                          # Local virtual environment (not versioned)
├── .gitignore
├── requirements.txt
└── README.md
```

## Current Status

**Status: In development — initial sprint**

| Stage | Status |
|-------|--------|
| EDA and class distribution analysis | Done [x] |
| Tokenization notebook | Done [x] |
| `src/preprocessing.py` module | Done [x] |
| Preprocessing unit tests | Done [x] |
| Transformer model fine-tuning | Pending [ ] |
| Final evaluation and benchmark metrics | Pending [ ] |

**Next steps:**

1. Structure the training script using the Hugging Face `Trainer` API
2. Define the evaluation protocol and baseline experiment
3. Document results and compare against the TweetEval leaderboard
