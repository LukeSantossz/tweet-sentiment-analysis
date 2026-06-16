# SPEC: fix(notebooks): load tokenization data from the Hub instead of a gitignored CSV

## Problem
`notebooks/02_tokenization.ipynb` reads `../data/sentiment.csv`, a file produced only by
notebook 01 and ignored by git, so notebook 02 cannot run on a fresh clone.

## Design Decision
Make notebook 02 self-contained by loading the TweetEval sentiment dataset directly from
the Hugging Face Hub (the same source notebook 01 uses), and remove notebook 01's now-unused
CSV export. Both notebooks then load from the Hub, with no shared on-disk artifact and no
hidden run-order dependency.

## Alternatives Considered
1. Commit `data/sentiment.csv` so notebook 02 can read it — rejected: versioning a derived
   ~6.8 MB dataset bloats the repo and contradicts the gitignored `data/` policy; the data
   is one `load_dataset` call away.
2. Change only notebook 02 to load from the Hub and keep notebook 01's CSV export — rejected:
   it leaves a dead side-effect that writes a gitignored file nothing reads; removing it
   makes the coupling fully gone.

## Scope
- Includes:
  - `notebooks/02_tokenization.ipynb`: import `datasets.load_dataset` and load the dataset
    from the Hub, replacing `pd.read_csv("../data/sentiment.csv")`.
  - `notebooks/01_eda.ipynb`: remove the unused CSV export (`sentiment_df` / `to_csv`) and the
    now-unused `all_data` variable.
- Does NOT include: notebook 03; any analysis logic or results; the `data/` gitignore policy.

## Acceptance Criteria
- nb02_self_contained: `notebooks/02_tokenization.ipynb` contains no read of
  `../data/sentiment.csv` and loads `tweet_eval`/`sentiment` from the Hub.
- nb01_no_csv_export: `notebooks/01_eda.ipynb` no longer writes `../data/sentiment.csv`.
- no_csv_coupling: no reference to `data/sentiment.csv` remains in any notebook.
- notebooks_valid: both notebooks remain valid JSON with unchanged cell counts.

## Reproducibility
- Scan: a search for `sentiment.csv` over `notebooks/` returns no matches.
- No code is executed in this cycle; running notebook 02 on a fresh clone now needs only
  network access to the Hub, the same as notebook 01 and the training script.

## Risks and Assumptions
- Assumption: notebook 02 only needs the `text` column (confirmed: it computes token counts
  over `dataset["text"]`); the Hub dataset provides it.
- Risk: notebook 02 now requires network access to the Hub at run time (previously it relied
  on a local CSV); this is acceptable and consistent with notebook 01 and `src/training.py`.
