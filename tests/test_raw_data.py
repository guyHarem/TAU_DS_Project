"""
Raw data density tests.

Checks that each combined raw symbol file is not sparse.

Recommended minimum density: 70% non-null values across non-time columns.
Usage:
    pytest tests/test_raw_data.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw_data"
MIN_DENSITY = 0.70
OHLCV_FIELDS = {"open", "high", "low", "close", "volume"}


def is_git_lfs_pointer(file_path: Path) -> bool:
    """Check whether a file is a Git LFS pointer file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            return first_line.startswith("version https://git-lfs.github.com")
    except OSError:
        return False


def get_raw_combined_files():
    """Return all combined raw-data CSV files (one per symbol/currency)."""
    return [
        p for p in sorted(RAW_DATA_DIR.glob("combined_*_data.csv"))
        if not is_git_lfs_pointer(p)
    ]


@pytest.fixture(scope="session")
def raw_files():
    files = get_raw_combined_files()
    if not files:
        pytest.skip("No combined raw data files available")
    return files


@pytest.mark.parametrize("csv_path", get_raw_combined_files())
def test_raw_file_has_only_exchange_ohlcv_columns(csv_path):
    """Raw files should contain only time and EXCHANGE:ohlcv columns."""
    df = pd.read_csv(csv_path)

    time_cols = [c for c in df.columns if c.lower() == "time"]
    assert len(time_cols) == 1, f"{csv_path.name}: expected exactly one 'time' column"

    value_cols = [c for c in df.columns if c.lower() != "time"]
    assert value_cols, f"{csv_path.name}: no exchange OHLCV columns found"

    for col in value_cols:
        assert ":" in col, f"{csv_path.name}: invalid column format '{col}', expected EXCHANGE:field"
        _, field = col.split(":", 1)
        assert field.lower() in OHLCV_FIELDS, (
            f"{csv_path.name}: unexpected field '{field}' in column '{col}'"
        )


@pytest.mark.parametrize("csv_path", get_raw_combined_files())
def test_each_exchange_has_full_ohlcv_set(csv_path):
    """Each exchange present in a file should include all 5 OHLCV fields."""
    df = pd.read_csv(csv_path)
    value_cols = [c for c in df.columns if c.lower() != "time"]

    fields_by_exchange = {}
    for col in value_cols:
        if ":" not in col:
            continue
        exchange, field = col.split(":", 1)
        fields_by_exchange.setdefault(exchange.upper(), set()).add(field.lower())

    assert fields_by_exchange, f"{csv_path.name}: no exchange columns detected"
    for exchange, fields in fields_by_exchange.items():
        assert fields == OHLCV_FIELDS, (
            f"{csv_path.name}: {exchange} fields={sorted(fields)}, "
            f"expected={sorted(OHLCV_FIELDS)}"
        )


@pytest.mark.parametrize("csv_path", get_raw_combined_files())
def test_raw_file_density(csv_path):
    """Each raw symbol file should have at least 70% non-null density over OHLCV columns."""
    df = pd.read_csv(csv_path)

    # Density is measured only over exchange OHLCV columns, excluding time.
    value_cols = [c for c in df.columns if c.lower() != "time"]
    assert value_cols, f"{csv_path.name}: no exchange OHLCV columns found"

    total_cells = len(df) * len(value_cols)
    assert total_cells > 0, f"{csv_path.name}: empty dataset"

    non_null_cells = int(df[value_cols].notna().sum().sum())
    density = non_null_cells / total_cells

    assert density >= MIN_DENSITY, (
        f"{csv_path.name}: density={density:.2%}, "
        f"required>={MIN_DENSITY:.0%}"
    )


@pytest.mark.parametrize("csv_path", get_raw_combined_files())
def test_raw_file_has_minimum_rows(csv_path):
    """Sanity check that each raw symbol file has a usable number of rows."""
    df = pd.read_csv(csv_path)
    assert len(df) >= 100, f"{csv_path.name}: too few rows ({len(df)})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
