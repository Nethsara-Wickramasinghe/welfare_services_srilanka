from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().split())


def prepare_housing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["district_name", "ds_division_name", "gn_division_name", "gn_division_code"]:
        df[col] = df[col].map(normalize_text)
    for col in ["district_code", "ds_division_code", "gn_division_number"]:
        df[col] = df[col].astype(str).str.strip()
    df["occupied_housing_units"] = pd.to_numeric(df["occupied_housing_units"], errors="coerce")
    grouped = (
        df.groupby(["district_name", "ds_division_name", "gn_division_name"], as_index=False)
        .agg(
            {
                "province_code": "first",
                "province_name": "first",
                "district_code": "first",
                "ds_division_code": "first",
                "gn_division_number": "first",
                "gn_division_code": "first",
                "occupied_housing_units": "median",
            }
        )
    )
    counts = (
        df.groupby(["district_name", "ds_division_name", "gn_division_name"])
        .size()
        .reset_index(name="housing_name_match_count")
    )
    grouped = grouped.merge(
        counts,
        on=["district_name", "ds_division_name", "gn_division_name"],
        how="left",
    )
    return grouped


def prepare_population(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["district_name", "ds_division_name", "gn_division_name"]:
        df[col] = df[col].map(normalize_text)
    for col in ["district_code", "ds_division_code", "gn_division_code"]:
        df[col] = df[col].astype("Int64").astype(str).str.strip()
    numeric_cols = [
        "population_total",
        "population_male",
        "population_female",
        "age_total",
        "age_0_14",
        "age_15_59",
        "age_60_64",
        "age_65_plus",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_context(population_csv: Path, housing_csv: Path, output_csv: Path) -> pd.DataFrame:
    population = prepare_population(pd.read_csv(population_csv))
    housing = prepare_housing(pd.read_csv(housing_csv))

    merged = population.merge(
        housing[
            [
                "province_code",
                "province_name",
                "district_code",
                "district_name",
                "ds_division_code",
                "ds_division_name",
                "gn_division_number",
                "gn_division_code",
                "gn_division_name",
                "occupied_housing_units",
            ]
        ],
        how="left",
        on=["district_name", "ds_division_name", "gn_division_name"],
        suffixes=("", "_housing"),
    )

    merged["occupied_housing_units"] = pd.to_numeric(merged["occupied_housing_units"], errors="coerce")
    merged["occupied_housing_units"] = merged["occupied_housing_units"].fillna(
        merged["occupied_housing_units"].median()
    )

    merged["children_ratio"] = merged["age_0_14"] / merged["population_total"].replace(0, np.nan)
    merged["working_age_ratio"] = merged["age_15_59"] / merged["population_total"].replace(0, np.nan)
    merged["elderly_ratio"] = (
        (merged["age_60_64"] + merged["age_65_plus"]) / merged["population_total"].replace(0, np.nan)
    )
    merged["old_age_ratio"] = merged["age_65_plus"] / merged["population_total"].replace(0, np.nan)
    merged["sex_ratio_male"] = merged["population_male"] / merged["population_total"].replace(0, np.nan)
    merged["sex_ratio_female"] = merged["population_female"] / merged["population_total"].replace(0, np.nan)
    merged["persons_per_housing_unit"] = merged["population_total"] / merged["occupied_housing_units"].replace(0, np.nan)
    merged["housing_units_log"] = np.log1p(merged["occupied_housing_units"])
    merged["population_log"] = np.log1p(merged["population_total"])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged official GN context dataset.")
    parser.add_argument("population_csv", type=Path)
    parser.add_argument("housing_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    merged = build_context(args.population_csv, args.housing_csv, args.output_csv)
    print(f"rows={len(merged)}")
    print(f"columns={len(merged.columns)}")
    print("missing_housing_after_merge=", int(merged["occupied_housing_units"].isna().sum()))
    print(merged.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
