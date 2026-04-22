from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FINAL_COLUMNS = [
    "province_code",
    "province_name",
    "district_code",
    "district_name",
    "ds_division_code",
    "ds_division_name",
    "gn_division_code",
    "gn_division_name",
    "gn_division_number",
    "population_total",
    "population_male",
    "population_female",
    "age_total",
    "age_0_14",
    "age_15_59",
    "age_60_64",
    "age_65_plus",
]


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def extract_population_table(xlsx_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(xlsx_path, sheet_name="Population", header=None)

    data = raw.iloc[4:, :17].copy()
    data.columns = FINAL_COLUMNS
    data = data.dropna(how="all")

    text_columns = [
        "province_name",
        "district_name",
        "ds_division_name",
        "gn_division_name",
    ]
    for col in text_columns:
        data[col] = data[col].map(normalize_text)

    numeric_columns = [
        "province_code",
        "district_code",
        "ds_division_code",
        "gn_division_code",
        "gn_division_number",
        "population_total",
        "population_male",
        "population_female",
        "age_total",
        "age_0_14",
        "age_15_59",
        "age_60_64",
        "age_65_plus",
    ]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(
        subset=[
            "province_code",
            "district_code",
            "ds_division_code",
            "gn_division_code",
            "gn_division_name",
            "population_total",
        ]
    )

    integer_columns = numeric_columns
    for col in integer_columns:
        data[col] = data[col].astype("Int64")

    return data.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract GN population workbook to CSV.")
    parser.add_argument("xlsx_path", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    df = extract_population_table(args.xlsx_path)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"rows={len(df)}")
    print(f"columns={len(df.columns)}")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
