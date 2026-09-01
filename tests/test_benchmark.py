"""Tests for the benchmark's Python preprocessor CLI and its comparison helpers."""

from pathlib import Path

import polars as pl
import pytest

from benchmarks.preprocessing_benchmark import median_of, validate_parity
from benchmarks.python_preprocessor import main, run


def _read_cleaned(path: Path) -> list[str]:
    return pl.read_parquet(path)["text_cleaned"].to_list()


def test_python_preprocessor_writes_text_cleaned_applying_the_model_contract(tmp_path: Path):
    # The model contract collapses @mentions and URLs but preserves case, hashtags and emoji.
    source = tmp_path / "in.csv"
    pl.DataFrame({"text": ["Check @john #AI 😊 https://example.com CAPS"]}).write_csv(source)
    out = tmp_path / "out.parquet"

    run(source, out, "text")

    assert _read_cleaned(out) == ["Check @user #AI 😊 http CAPS"]


def test_python_preprocessor_maps_null_text_to_empty_string(tmp_path: Path):
    # Matches the Rust CLI's null policy, so a null never aborts a batch and the
    # two outputs stay comparable row for row.
    source = tmp_path / "in.parquet"
    pl.DataFrame({"text": ["x", None, "y"]}).write_parquet(source)
    out = tmp_path / "out.parquet"

    run(source, out, "text")

    assert _read_cleaned(out) == ["x", "", "y"]


def test_python_preprocessor_preserves_row_count_and_order(tmp_path: Path):
    source = tmp_path / "in.csv"
    rows = [f"row {index} @user{index}" for index in range(50)]
    pl.DataFrame({"text": rows}).write_csv(source)
    out = tmp_path / "out.parquet"

    run(source, out, "text")

    cleaned = _read_cleaned(out)
    assert len(cleaned) == 50
    assert cleaned[0].startswith("row 0 ")
    assert cleaned[49].startswith("row 49 ")


def test_python_preprocessor_keeps_the_original_columns(tmp_path: Path):
    # The Rust CLI appends text_cleaned to the existing frame; parity requires the same shape.
    source = tmp_path / "in.csv"
    pl.DataFrame({"text": ["@a"], "label": ["joy"]}).write_csv(source)
    out = tmp_path / "out.parquet"

    run(source, out, "text")

    assert pl.read_parquet(out).columns == ["text", "label", "text_cleaned"]


def test_python_preprocessor_reads_parquet_as_well_as_csv(tmp_path: Path):
    source = tmp_path / "in.parquet"
    pl.DataFrame({"text": ["@a http://x.co"]}).write_parquet(source)
    out = tmp_path / "out.parquet"

    run(source, out, "text")

    assert _read_cleaned(out) == ["@user http"]


def test_python_preprocessor_rejects_an_unsupported_extension(tmp_path: Path):
    source = tmp_path / "in.json"
    source.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV or Parquet"):
        run(source, tmp_path / "out.parquet", "text")


def test_python_preprocessor_missing_column_raises(tmp_path: Path):
    source = tmp_path / "in.csv"
    pl.DataFrame({"body": ["x"]}).write_csv(source)

    with pytest.raises(ValueError, match="text"):
        run(source, tmp_path / "out.parquet", "text")


def test_python_preprocessor_main_wires_the_cli_arguments(tmp_path: Path):
    source = tmp_path / "in.csv"
    pl.DataFrame({"body": ["@a"]}).write_csv(source)
    out = tmp_path / "out.parquet"

    main(["-i", str(source), "-o", str(out), "-c", "body"])

    assert _read_cleaned(out) == ["@user"]


def test_validate_parity_compares_two_parquet_files(tmp_path: Path):
    left, right = tmp_path / "l.parquet", tmp_path / "r.parquet"
    pl.DataFrame({"text_cleaned": ["a", "b", "c"]}).write_parquet(left)
    pl.DataFrame({"text_cleaned": ["a", "b", "c"]}).write_parquet(right)

    assert validate_parity(left, right) == (True, 0)


def test_validate_parity_counts_mismatches(tmp_path: Path):
    left, right = tmp_path / "l.parquet", tmp_path / "r.parquet"
    pl.DataFrame({"text_cleaned": ["a", "b", "c"]}).write_parquet(left)
    pl.DataFrame({"text_cleaned": ["a", "X", "Y"]}).write_parquet(right)

    assert validate_parity(left, right) == (False, 2)


def test_validate_parity_reports_a_row_count_difference(tmp_path: Path):
    left, right = tmp_path / "l.parquet", tmp_path / "r.parquet"
    pl.DataFrame({"text_cleaned": ["a", "b", "c"]}).write_parquet(left)
    pl.DataFrame({"text_cleaned": ["a"]}).write_parquet(right)

    ok, mismatches = validate_parity(left, right)
    assert ok is False
    assert mismatches == 2


def test_validate_parity_unreadable_output_returns_none_not_sentinel(tmp_path: Path):
    # A missing/unreadable output must not surface a count; None means "could not
    # compare", distinct from 0, which means "compared, fully matching".
    left = tmp_path / "l.parquet"
    pl.DataFrame({"text_cleaned": ["a"]}).write_parquet(left)

    ok, mismatch_count = validate_parity(left, tmp_path / "missing.parquet")

    assert ok is False
    assert mismatch_count is None


def test_median_of_returns_the_middle_measurement():
    assert median_of([3.0, 1.0, 2.0]) == 2.0


def test_median_of_averages_the_middle_pair_for_an_even_count():
    assert median_of([4.0, 1.0, 3.0, 2.0]) == pytest.approx(2.5)


def test_median_of_a_single_measurement_is_that_measurement():
    assert median_of([1.5]) == 1.5
