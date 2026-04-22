from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from pypdf import PdfReader


PROVINCE_NAMES = [
    "North Central",
    "North Western",
    "Sabaragamuwa",
    "Central",
    "Eastern",
    "Northern",
    "Southern",
    "Uva",
    "Western",
]

DISTRICT_NAMES = [
    "Nuwara Eliya",
    "Anuradhapura",
    "Polonnaruwa",
    "Kurunegala",
    "Puttalam",
    "Kegalle",
    "Ratnapura",
    "Trincomalee",
    "Batticaloa",
    "Mullaitivu",
    "Kilinochchi",
    "Monaragala",
    "Moneragala",
    "Hambantota",
    "Matale",
    "Kandy",
    "Galle",
    "Matara",
    "Ampara",
    "Vavuniya",
    "Mannar",
    "Jaffna",
    "Badulla",
    "Colombo",
    "Gampaha",
    "Kalutara",
]

HEADER_FRAGMENTS = {
    "GN_Division",
    "Occupied",
    "Housing",
    "Province",
    "District",
    "DS_Division",
    "Code",
    "Name",
    "Number",
    "Units",
}

CODE_PATTERN = re.compile(r"^[A-Za-z0-9/'-]*\d[A-Za-z0-9/'-]*$")
GN_NUMBER_PATTERN = re.compile(r"^\d{3}$")


def match_name(tokens: list[str], candidates: list[str]) -> tuple[str, int]:
    lowered = [token.lower() for token in tokens]
    for candidate in candidates:
        parts = candidate.lower().split()
        length = len(parts)
        if lowered[:length] == parts:
            return candidate, length
    raise ValueError(f"Could not match name from tokens: {' '.join(tokens[:6])}")


def clean_line(line: str) -> str:
    line = " ".join(line.split()).strip()
    # Some PDF rows merge the end of the DS division name with the GN number.
    line = re.sub(r"([A-Za-z\)])(\d{3})(\s)", r"\1 \2\3", line)
    # Some GN codes appear split as "103 E" instead of "103E".
    line = re.sub(r"\b(\d{2,4})\s+([A-Z])\s+((?:\d[\d,]*|-))$", r"\1\2 \3", line)
    return line


def is_probable_data_line(line: str) -> bool:
    if not line or not line[0].isdigit():
        return False
    if any(fragment in line for fragment in HEADER_FRAGMENTS):
        return False
    return True


def parse_row(line: str) -> dict[str, str | int | None]:
    tokens = line.split()
    province_code = int(tokens[0])

    province_name, province_len = match_name(tokens[1:], PROVINCE_NAMES)
    idx = 1 + province_len

    district_code = int(tokens[idx])
    idx += 1

    district_name, district_len = match_name(tokens[idx:], DISTRICT_NAMES)
    idx += district_len

    ds_division_code = tokens[idx]
    idx += 1

    occupied_housing_units = None
    if tokens[-1] != "-":
        occupied_housing_units = int(tokens[-1].replace(",", ""))

    trailing_tokens = tokens[idx:-1]
    gn_number_index = next(
        i for i, token in enumerate(trailing_tokens) if GN_NUMBER_PATTERN.fullmatch(token)
    )

    ds_division_name = " ".join(trailing_tokens[:gn_number_index])
    gn_division_number = trailing_tokens[gn_number_index]
    gn_tokens = trailing_tokens[gn_number_index + 1 :]

    gn_division_code = ""
    if gn_tokens and CODE_PATTERN.fullmatch(gn_tokens[-1]):
        gn_division_code = gn_tokens[-1]
        gn_division_name = " ".join(gn_tokens[:-1]).strip()
    else:
        gn_division_name = " ".join(gn_tokens).strip()

    if not ds_division_name or not gn_division_name:
        raise ValueError(f"Incomplete parse for line: {line}")

    return {
        "province_code": province_code,
        "province_name": province_name,
        "district_code": district_code,
        "district_name": district_name,
        "ds_division_code": ds_division_code,
        "ds_division_name": ds_division_name,
        "gn_division_number": gn_division_number,
        "gn_division_code": gn_division_code,
        "gn_division_name": gn_division_name,
        "occupied_housing_units": occupied_housing_units,
    }


def extract_rows(pdf_path: Path) -> tuple[list[dict[str, str | int | None]], list[str]]:
    reader = PdfReader(str(pdf_path))
    rows: list[dict[str, str | int | None]] = []
    failures: list[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        cleaned_lines = [clean_line(raw_line) for raw_line in text.splitlines()]
        i = 0
        while i < len(cleaned_lines):
            line = cleaned_lines[i]
            if not is_probable_data_line(line):
                i += 1
                continue
            try:
                rows.append(parse_row(line))
                i += 1
                continue
            except Exception:
                pass

            if i + 1 < len(cleaned_lines):
                combined = f"{line} {cleaned_lines[i + 1]}".strip()
                try:
                    rows.append(parse_row(combined))
                    i += 2
                    continue
                except Exception:
                    failures.append(line)
                    i += 1
                    continue

            failures.append(line)
            i += 1
    return rows, failures


def write_csv(rows: list[dict[str, str | int | None]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract GN housing rows from PDF to CSV.")
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    rows, failures = extract_rows(args.pdf_path)
    write_csv(rows, args.output_csv)

    print(f"rows_extracted={len(rows)}")
    print(f"rows_failed={len(failures)}")
    if failures:
        print("sample_failures=")
        for line in failures[:10]:
            print(line)


if __name__ == "__main__":
    main()
