"""Tests for the preprocessing benchmark helpers."""

from pathlib import Path

from benchmarks.preprocessing_benchmark import validate_parity


def test_validate_parity_unreadable_output_returns_none_not_sentinel(tmp_path: Path):
    # A missing/unreadable Rust output must not surface -1 as a literal mismatch count;
    # None means "could not compare", distinct from 0 ("compared, fully matching").
    ok, mismatch_count = validate_parity(["a", "b"], tmp_path / "missing.parquet")

    assert ok is False
    assert mismatch_count is None
