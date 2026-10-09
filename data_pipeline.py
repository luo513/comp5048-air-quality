from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SELECTED_ATTRIBUTES = [
    "DateTime",
    "CO(GT)",
    "PT08.S1(CO)",
    "C6H6(GT)",
    "NOx(GT)",
    "NO2(GT)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]


def load_and_clean(source: Path) -> pd.DataFrame:
    """Load the UCI file without modifying or imputing missing observations."""
    raw = pd.read_csv(source, sep=";", decimal=",")
    raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all")

    raw["DateTime"] = pd.to_datetime(
        raw["Date"].astype(str) + " " + raw["Time"].astype(str),
        format="%d/%m/%Y %H.%M.%S",
        errors="raise",
    )

    numeric_attributes = SELECTED_ATTRIBUTES[1:]
    raw[numeric_attributes] = raw[numeric_attributes].replace(-200, pd.NA)

    clean = raw[SELECTED_ATTRIBUTES].copy()
    clean.insert(1, "Hour", clean["DateTime"].dt.hour)
    clean.insert(2, "Weekday", clean["DateTime"].dt.day_name())
    clean.insert(3, "Month", clean["DateTime"].dt.to_period("M").astype(str))
    return clean.sort_values("DateTime").reset_index(drop=True)


def quality_summary(clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in SELECTED_ATTRIBUTES:
        missing = int(clean[column].isna().sum())
        rows.append(
            {
                "attribute": column,
                "missing_count": missing,
                "missing_pct": missing / len(clean),
                "non_missing_count": int(clean[column].notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("AirQualityUCI.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    clean = load_and_clean(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output_dir / "air_quality_clean.csv", index=False)
    quality_summary(clean).to_csv(
        args.output_dir / "quality_summary.csv", index=False, float_format="%.6f"
    )

    complete = int(clean[SELECTED_ATTRIBUTES[1:]].notna().all(axis=1).sum())
    print(f"Rows: {len(clean):,}")
    print(f"Selected attributes: {len(SELECTED_ATTRIBUTES)}")
    print(f"Complete rows across numeric attributes: {complete:,}")
    print("Missing values were retained as blank; no imputation was performed.")


if __name__ == "__main__":
    main()
