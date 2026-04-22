# GN Housing PDF Integration

## Source

- Official PDF: `d:\MyStuff\Downloads\GN_housing_pdf.pdf`
- Extractor script: `scripts/extract_gn_housing_pdf.py`
- Extracted CSV: `data/official/gn_housing_units.csv`

## Extracted Schema

The CSV contains one row per GN division with these columns:

- `province_code`
- `province_name`
- `district_code`
- `district_name`
- `ds_division_code`
- `ds_division_name`
- `gn_division_number`
- `gn_division_code`
- `gn_division_name`
- `occupied_housing_units`

## How To Use In The Research Dataset

This dataset is best used as an area-level context table, not as the main welfare training dataset by itself.

Recommended joins:

1. Join household/application records to this table using the most specific location keys available.
2. Prefer this order when matching:
   - `district_name` + `ds_division_name` + `gn_division_name`
   - `district_code` + `ds_division_code` + `gn_division_code`
   - `district_code` + `ds_division_code` + `gn_division_number`

## Recommended Derived Features

Add these area features into the ML pipeline:

- `occupied_housing_units`
- `gn_has_missing_code`
- `housing_units_log = log1p(occupied_housing_units)`
- `housing_size_band`:
  - small: `< 500`
  - medium: `500-1499`
  - large: `1500-2999`
  - very_large: `>= 3000`

If population or land-area data is added later, also derive:

- `housing_units_per_capita`
- `housing_density_proxy`

## Research Use

This table supports:

- GN-level mapping and regional analysis
- area-context features for service recommendation models
- district / DS / GN aggregation for dashboards
- vulnerability scoring when combined with socioeconomic data

## Important Note

This source does not contain welfare labels or direct household vulnerability attributes such as income, disability, unemployment, or elderly counts. It should be merged with household-level or census-style data for model training.
