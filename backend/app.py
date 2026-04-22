from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

GN_CONTEXT_PATH = os.path.join(PROJECT_ROOT, "data", "research", "gn_official_context_dataset.csv")
PROGRAM_CATALOG_PATH = os.path.join(BASE_DIR, "program_catalog.json")
CITIZEN_SERVICE_MODEL_PATH = os.path.join(BASE_DIR, "models", "citizen_service_model.pkl")
CITIZEN_PRIORITY_MODEL_PATH = os.path.join(BASE_DIR, "models", "citizen_priority_model.pkl")
CITIZEN_OUTCOME_MODEL_PATH = os.path.join(BASE_DIR, "models", "citizen_outcome_model.pkl")
CITIZEN_MODEL_METADATA_PATH = os.path.join(BASE_DIR, "models", "citizen_model_metadata.json")

INCOME_RANGE_MAPPING = {
    "below-25000": 15000,
    "25000-50000": 37500,
    "50000-100000": 75000,
    "100000-200000": 150000,
    "above-200000": 250000,
}

LOCATION_REQUIRED_FIELDS = [
    "district_name",
    "ds_division_name",
    "gn_division_name",
    "family_size",
    "income_range",
    "employment_status",
    "has_elderly",
    "has_disabled_member",
    "has_chronic_illness",
    "recent_disaster_impact",
    "food_insecurity_score",
    "healthcare_access_score",
]

VALID_INCOME_RANGES = set(INCOME_RANGE_MAPPING.keys())
VALID_EMPLOYMENT_STATUSES = {
    "employed",
    "self-employed",
    "unemployed",
    "retired",
    "student",
    "unable-to-work",
}

NEXT_STEP_LIBRARY = {
    "Food Assistance": "Carry NIC and income proof when visiting the Divisional Secretariat.",
    "Health Support": "Bring clinic records, disability certificates, or chronic illness documents if available.",
    "Elderly Care": "Bring documents related to elderly dependents, such as NIC copies or medical notes.",
    "Disaster Relief": "Bring any proof of disaster impact, damage photos, or local authority confirmation if available.",
}

SERVICE_PROGRAM_MAP = {
    "Food Assistance": ["aswesuma"],
    "Health Support": ["disability_assistance", "kidney_disease_assistance", "free_government_healthcare"],
    "Elderly Care": ["elderly_assistance", "helpage_referral"],
    "Disaster Relief": ["disaster_relief", "sarvodaya_referral"],
}

MIN_SERVICE_CONFIDENCE = 0.35

gn_context_df = pd.read_csv(GN_CONTEXT_PATH)
gn_context_df[["district_name", "ds_division_name", "gn_division_name"]] = gn_context_df[
    ["district_name", "ds_division_name", "gn_division_name"]
].fillna("").apply(lambda col: col.astype(str).str.strip().str.lower())
gn_context_df["display_district_name"] = gn_context_df["district_name"].str.title()
gn_context_df["display_ds_division_name"] = gn_context_df["ds_division_name"].str.title()
gn_context_df["display_gn_division_name"] = gn_context_df["gn_division_name"].str.title()
gn_context_df["elderly_ratio"] = pd.to_numeric(gn_context_df["elderly_ratio"], errors="coerce").fillna(0.0)
gn_context_df["children_ratio"] = pd.to_numeric(gn_context_df["children_ratio"], errors="coerce").fillna(0.0)
gn_context_df["persons_per_housing_unit"] = pd.to_numeric(
    gn_context_df["persons_per_housing_unit"], errors="coerce"
).fillna(0.0)
gn_context_df["population_total"] = pd.to_numeric(gn_context_df["population_total"], errors="coerce").fillna(0.0)
gn_context_df["occupied_housing_units"] = pd.to_numeric(
    gn_context_df["occupied_housing_units"], errors="coerce"
).fillna(0.0)
gn_context_df["housing_units_log"] = pd.to_numeric(gn_context_df["housing_units_log"], errors="coerce").fillna(0.0)
gn_context_df["population_log"] = pd.to_numeric(gn_context_df["population_log"], errors="coerce").fillna(0.0)
if "housing_name_match_count" in gn_context_df.columns:
    gn_context_df["housing_name_match_count"] = pd.to_numeric(
        gn_context_df["housing_name_match_count"], errors="coerce"
    ).fillna(1).astype(int)
else:
    gn_context_df["housing_name_match_count"] = 1

location_cache: dict[str, Any] | None = None
program_catalog: dict[str, Any] | None = None
citizen_service_model = None
citizen_priority_model = None
citizen_outcome_model = None
citizen_model_metadata: dict[str, Any] | None = None


def normalize_location(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def validate_assessment_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    try:
        family_size = int(payload.get("family_size"))
        if family_size < 1 or family_size > 20:
            errors.append("Family size must be between 1 and 20.")
    except (TypeError, ValueError):
        errors.append("Family size must be a whole number.")

    income_range = payload.get("income_range")
    if income_range not in VALID_INCOME_RANGES:
        errors.append("Monthly income range is invalid.")

    employment_status = normalize_location(payload.get("employment_status"))
    if employment_status not in VALID_EMPLOYMENT_STATUSES:
        errors.append("Employment status is invalid.")

    for field_name, label in [
        ("has_elderly", "Elderly household flag"),
        ("has_disabled_member", "Disabled household flag"),
        ("has_chronic_illness", "Chronic illness flag"),
        ("recent_disaster_impact", "Recent disaster flag"),
    ]:
        try:
            value = int(payload.get(field_name))
            if value not in {0, 1}:
                errors.append(f"{label} must be either 0 or 1.")
        except (TypeError, ValueError):
            errors.append(f"{label} must be either 0 or 1.")

    for field_name, label in [
        ("food_insecurity_score", "Food insecurity score"),
        ("healthcare_access_score", "Healthcare access score"),
    ]:
        try:
            value = int(payload.get(field_name))
            if value < 1 or value > 5:
                errors.append(f"{label} must be between 1 and 5.")
        except (TypeError, ValueError):
            errors.append(f"{label} must be a whole number between 1 and 5.")

    return errors


def load_optional_citizen_models() -> None:
    global citizen_service_model, citizen_priority_model, citizen_outcome_model, citizen_model_metadata
    model_paths = [CITIZEN_SERVICE_MODEL_PATH, CITIZEN_PRIORITY_MODEL_PATH, CITIZEN_MODEL_METADATA_PATH]
    if all(os.path.exists(path) for path in model_paths):
        citizen_service_model = joblib.load(CITIZEN_SERVICE_MODEL_PATH)
        citizen_priority_model = joblib.load(CITIZEN_PRIORITY_MODEL_PATH)
        if os.path.exists(CITIZEN_OUTCOME_MODEL_PATH):
            citizen_outcome_model = joblib.load(CITIZEN_OUTCOME_MODEL_PATH)
        with open(CITIZEN_MODEL_METADATA_PATH, "r", encoding="utf-8") as metadata_file:
            citizen_model_metadata = json.load(metadata_file)


def load_program_catalog() -> None:
    global program_catalog
    if os.path.exists(PROGRAM_CATALOG_PATH):
        with open(PROGRAM_CATALOG_PATH, "r", encoding="utf-8") as catalog_file:
            program_catalog = json.load(catalog_file)
    else:
        program_catalog = {"metadata": {}, "programs": []}


load_optional_citizen_models()
load_program_catalog()


def build_location_tree() -> dict[str, Any]:
    districts = []
    for district_key, district_df in gn_context_df.groupby("district_name", sort=True):
        if not district_key:
            continue
        ds_entries = []
        for ds_key, ds_df in district_df.groupby("ds_division_name", sort=True):
            if not ds_key:
                continue
            gn_entries = []
            for _, gn_row in ds_df.sort_values("display_gn_division_name").iterrows():
                gn_entries.append(
                    {
                        "name": gn_row["display_gn_division_name"],
                        "value": gn_row["gn_division_name"],
                        "population_total": int(gn_row["population_total"]),
                        "occupied_housing_units": int(round(gn_row["occupied_housing_units"])),
                    }
                )
            ds_entries.append(
                {
                    "name": ds_df["display_ds_division_name"].iloc[0],
                    "value": ds_key,
                    "gn_divisions": gn_entries,
                }
            )
        districts.append(
            {
                "name": district_df["display_district_name"].iloc[0],
                "value": district_key,
                "ds_divisions": ds_entries,
            }
        )
    return {"districts": districts}


def get_location_tree() -> dict[str, Any]:
    global location_cache
    if location_cache is None:
        location_cache = build_location_tree()
    return location_cache


def get_gn_context(district_name: str, ds_division_name: str, gn_division_name: str) -> pd.Series | None:
    district_key = normalize_location(district_name)
    ds_key = normalize_location(ds_division_name)
    gn_key = normalize_location(gn_division_name)

    matches = gn_context_df[
        (gn_context_df["district_name"] == district_key)
        & (gn_context_df["ds_division_name"] == ds_key)
        & (gn_context_df["gn_division_name"] == gn_key)
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def build_citizen_feature_row(payload: dict[str, Any], context_row: pd.Series) -> dict[str, Any]:
    return {
        "district_code": int(context_row["district_code"]),
        "ds_division_code": int(context_row["ds_division_code"]),
        "population_total": float(context_row["population_total"]),
        "occupied_housing_units": float(context_row["occupied_housing_units"]),
        "children_ratio": float(context_row["children_ratio"]),
        "elderly_ratio": float(context_row["elderly_ratio"]),
        "persons_per_housing_unit": float(context_row["persons_per_housing_unit"]),
        "housing_units_log": float(context_row["housing_units_log"]),
        "population_log": float(context_row["population_log"]),
        "family_size": max(int(payload["family_size"]), 1),
        "income_value": INCOME_RANGE_MAPPING.get(payload["income_range"], 50000),
        "employment_status": normalize_location(payload["employment_status"]),
        "has_elderly": int(payload["has_elderly"]),
        "has_disabled_member": int(payload["has_disabled_member"]),
        "has_chronic_illness": int(payload["has_chronic_illness"]),
        "recent_disaster_impact": int(payload["recent_disaster_impact"]),
        "food_insecurity_score": max(1, min(int(payload["food_insecurity_score"]), 5)),
        "healthcare_access_score": max(1, min(int(payload["healthcare_access_score"]), 5)),
    }


def build_reason_breakdown(
    payload: dict[str, Any],
    context_row: pd.Series,
    recommended_services: list[dict[str, Any]] | None = None,
    matched_programs: list[dict[str, Any]] | None = None,
    priority_level: str | None = None,
) -> tuple[list[str], list[str]]:
    family_size = max(int(payload["family_size"]), 1)
    income_value = INCOME_RANGE_MAPPING.get(payload["income_range"], 50000)
    income_per_capita = income_value / family_size
    food_insecurity_score = max(1, min(int(payload["food_insecurity_score"]), 5))
    employment_status = normalize_location(payload.get("employment_status"))
    has_elderly = int(payload["has_elderly"]) == 1
    has_disabled_member = int(payload["has_disabled_member"]) == 1
    has_chronic_illness = int(payload["has_chronic_illness"]) == 1
    recent_disaster_impact = int(payload["recent_disaster_impact"]) == 1

    poverty_score = max(0.0, min(1.0, 1 - min(income_per_capita / 50000, 1)))
    household_factors = []
    contextual_factors = []

    if poverty_score >= 0.6:
        household_factors.append("Low income range relative to household size")
    if food_insecurity_score >= 4:
        household_factors.append("High reported food insecurity")
    if has_disabled_member:
        household_factors.append("A disabled household member increases health support need")
    if has_chronic_illness:
        household_factors.append("Chronic illness raises the need for healthcare support")
    if has_elderly:
        household_factors.append("The household includes an elderly family member who may need support")
    if recent_disaster_impact:
        household_factors.append("Recent disaster impact suggests urgent relief support")
    if employment_status == "self-employed":
        household_factors.append("Self-employment status may match livelihood or finance referral pathways")
    elif employment_status in {"unemployed", "unable-to-work"}:
        household_factors.append("Current employment status may increase eligibility for income or livelihood support")

    if context_row["elderly_ratio"] >= 0.14:
        contextual_factors.append("The selected GN division has a relatively high elderly population share")
    if context_row["persons_per_housing_unit"] >= 4:
        contextual_factors.append("The selected GN division has relatively high housing pressure")

    recommended_services = recommended_services or []
    matched_programs = matched_programs or []

    if not household_factors and not recommended_services and not matched_programs:
        household_factors = [
            "Your household inputs do not indicate a strong direct welfare need at this time.",
            "No service or programme trigger was strongly matched from the current household details.",
        ]
        location_reasons = [
            "Location context is shown below for reference, but area-level indicators alone did not create a direct programme match."
        ]
        return household_factors[:4], location_reasons[:3]

    if not household_factors and priority_level == "Low":
        household_factors.append("Your household inputs currently indicate low immediate welfare vulnerability")

    location_reasons = []
    if contextual_factors and (recommended_services or priority_level in {"Medium", "High"}):
        location_reasons.extend(contextual_factors)
    elif contextual_factors and not household_factors:
        location_reasons.append("Location context was considered together with your household details when generating this result")

    if not household_factors:
        household_factors.append("The selected location and household inputs indicate moderate welfare vulnerability")
    return household_factors[:4], location_reasons[:3]


def build_top_factors(
    payload: dict[str, Any],
    context_row: pd.Series,
    recommended_services: list[dict[str, Any]] | None = None,
    matched_programs: list[dict[str, Any]] | None = None,
    priority_level: str | None = None,
) -> list[str]:
    household_reasons, location_reasons = build_reason_breakdown(
        payload,
        context_row,
        recommended_services,
        matched_programs,
        priority_level,
    )
    return (household_reasons + location_reasons)[:4]


def build_next_steps(recommended_services: list[dict[str, Any]], matched_programs: list[dict[str, Any]] | None = None) -> list[str]:
    next_steps = [
        "Visit your nearest Divisional Secretariat or welfare office for official verification.",
        "Keep your NIC and proof of address ready before applying.",
    ]
    if matched_programs:
        for program in matched_programs:
            for step in program.get("next_steps", []):
                if step not in next_steps:
                    next_steps.append(step)
    for service in recommended_services:
        guidance = NEXT_STEP_LIBRARY.get(service["service"])
        if guidance and guidance not in next_steps:
            next_steps.append(guidance)
    return next_steps[:6]


def split_program_matches(matched_programs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    direct_welfare_programs = []
    referral_programs = []
    for program in matched_programs:
        if program.get("group") == "government_primary":
            direct_welfare_programs.append(program)
        else:
            referral_programs.append(program)
    return direct_welfare_programs, referral_programs


def build_outcome_summary(
    payload: dict[str, Any],
    recommended_services: list[dict[str, Any]],
    direct_welfare_programs: list[dict[str, Any]],
    referral_programs: list[dict[str, Any]],
) -> str:
    employment_status = normalize_location(payload.get("employment_status"))

    if direct_welfare_programs:
        return "Your household details suggest a meaningful welfare support match, and the programmes below are the strongest direct options identified."
    if recommended_services and referral_programs:
        return "Your household details suggest some support needs, but the strongest current matches are referral pathways rather than direct welfare programmes."
    if referral_programs:
        if employment_status == "self-employed":
            return "No strong direct welfare match was identified, but your self-employment status matched a livelihood or finance referral that may still be useful."
        if employment_status in {"unemployed", "unable-to-work"}:
            return "No strong direct welfare match was identified, but your current employment situation matched referral pathways that may help with livelihood or support access."
        return "No strong direct welfare match was identified, but some referral pathways were found based on your current household details."
    return "Your household details do not currently show a strong direct welfare or programme match."


def determine_outcome_type(
    recommended_services: list[dict[str, Any]],
    direct_welfare_programs: list[dict[str, Any]],
    referral_programs: list[dict[str, Any]],
    priority_level: str,
) -> str:
    if direct_welfare_programs:
        return "direct_welfare_match"
    if recommended_services and referral_programs:
        return "referral_only_with_service_signal"
    if referral_programs:
        return "referral_only"
    if priority_level == "High":
        return "high_priority_no_programme_match"
    if priority_level == "Medium":
        return "moderate_priority_no_programme_match"
    return "no_strong_match"


def service_passes_household_guardrail(payload: dict[str, Any], service_name: str) -> bool:
    income_range = payload.get("income_range")
    employment_status = normalize_location(payload.get("employment_status"))
    has_elderly = int(payload.get("has_elderly", 0)) == 1
    has_disabled_member = int(payload.get("has_disabled_member", 0)) == 1
    has_chronic_illness = int(payload.get("has_chronic_illness", 0)) == 1
    recent_disaster_impact = int(payload.get("recent_disaster_impact", 0)) == 1
    food_insecurity_score = max(1, min(int(payload.get("food_insecurity_score", 1)), 5))
    healthcare_access_score = max(1, min(int(payload.get("healthcare_access_score", 5)), 5))

    if service_name == "Elderly Care":
        return has_elderly
    if service_name == "Disaster Relief":
        return recent_disaster_impact
    if service_name == "Health Support":
        return has_disabled_member or has_chronic_illness or healthcare_access_score <= 2
    if service_name == "Food Assistance":
        return (
            income_range in {"below-25000", "25000-50000"}
            or employment_status in {"unemployed", "unable-to-work"}
            or food_insecurity_score >= 4
        )
    return True


def apply_service_guardrails(payload: dict[str, Any], recommended_services: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        service
        for service in recommended_services
        if service_passes_household_guardrail(payload, service["service"])
    ]


def get_service_threshold(service_label: str) -> float:
    if citizen_model_metadata and "service_thresholds" in citizen_model_metadata:
        return float(citizen_model_metadata["service_thresholds"].get(service_label, MIN_SERVICE_CONFIDENCE))
    return MIN_SERVICE_CONFIDENCE


def normalize_payload_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_location(value)
    return value


def evaluate_trigger_condition(payload: dict[str, Any], condition: dict[str, Any]) -> bool:
    field = condition.get("field")
    operator = condition.get("operator")
    expected = condition.get("value")
    actual = normalize_payload_value(payload.get(field))

    if operator == "==":
        return actual == expected
    if operator == ">=":
        return float(actual) >= float(expected)
    if operator == "<=":
        return float(actual) <= float(expected)
    if operator == "in":
        normalized_expected = [normalize_payload_value(item) for item in expected]
        return actual in normalized_expected
    return False


def program_trigger_matches(payload: dict[str, Any], program: dict[str, Any]) -> tuple[bool, list[str]]:
    trigger_logic = program.get("trigger_logic", {})
    reasons: list[str] = []

    any_conditions = trigger_logic.get("recommended_when_any", [])
    if any_conditions:
        any_matched = any(evaluate_trigger_condition(payload, condition) for condition in any_conditions)
        if any_matched:
            reasons.append("Matches household trigger conditions")
        else:
            return False, []

    all_conditions = trigger_logic.get("recommended_when_all", [])
    if all_conditions:
        all_matched = all(evaluate_trigger_condition(payload, condition) for condition in all_conditions)
        if all_matched:
            reasons.append("Matches combined household conditions")
        else:
            return False, []

    if not any_conditions and not all_conditions:
        return False, []

    return True, reasons


def summarize_program(program: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    return {
        "id": program["id"],
        "display_name": program["display_name"],
        "group": program["group"],
        "category": program["category"],
        "authority": program["authority"],
        "official_url": program.get("official_url"),
        "contact": program.get("contact", {}),
        "summary": program.get("summary", ""),
        "benefit_note": program.get("benefit_note", ""),
        "next_steps": program.get("next_steps", []),
        "documents": program.get("documents", []),
        "source_links": program.get("source_links", []),
        "recommendation_reasons": reasons,
    }


def match_programs_to_assessment(payload: dict[str, Any], recommended_services: list[dict[str, Any]]) -> list[dict[str, Any]]:
    catalog_programs = (program_catalog or {}).get("programs", [])
    catalog_by_id = {program["id"]: program for program in catalog_programs}
    matched: list[dict[str, Any]] = []
    added_ids: set[str] = set()

    for service in recommended_services:
        for program_id in SERVICE_PROGRAM_MAP.get(service["service"], []):
            program = catalog_by_id.get(program_id)
            if not program or program_id in added_ids:
                continue
            matches, trigger_reasons = program_trigger_matches(payload, program)
            if not matches:
                continue
            reasons = [f"Aligned with the recommended service: {service['service']}"]
            reasons.extend(trigger_reasons)
            matched.append(summarize_program(program, reasons))
            added_ids.add(program_id)

    for program in catalog_programs:
        if program["id"] in added_ids:
            continue
        matches, trigger_reasons = program_trigger_matches(payload, program)
        if not matches:
            continue
        matched.append(summarize_program(program, trigger_reasons))
        added_ids.add(program["id"])

    group_priority = {"government_primary": 0, "government_secondary": 1, "ngo_secondary": 2}
    matched.sort(key=lambda item: (group_priority.get(item["group"], 9), item["display_name"]))
    return matched[:6]


def scoring_fallback_assessment(payload: dict[str, Any], context_row: pd.Series) -> dict[str, Any]:
    feature_row = build_citizen_feature_row(payload, context_row)
    family_size = feature_row["family_size"]
    income_value = feature_row["income_value"]
    income_per_capita = income_value / family_size
    food_insecurity_score = feature_row["food_insecurity_score"]
    healthcare_access_score = feature_row["healthcare_access_score"]
    has_elderly = feature_row["has_elderly"] == 1
    has_disabled_member = feature_row["has_disabled_member"] == 1
    has_chronic_illness = feature_row["has_chronic_illness"] == 1
    recent_disaster_impact = feature_row["recent_disaster_impact"] == 1
    employment_status = feature_row["employment_status"]

    poverty_score = max(0.0, min(1.0, 1 - min(income_per_capita / 50000, 1)))
    employment_risk = 1.0 if employment_status in {"unemployed", "unable-to-work"} else 0.4 if employment_status == "retired" else 0.2
    healthcare_access_inverse = 1 - ((healthcare_access_score - 1) / 4)
    food_insecurity_norm = (food_insecurity_score - 1) / 4

    service_scores = {
        "Food Assistance": (
            poverty_score * 0.45
            + food_insecurity_norm * 0.30
            + employment_risk * 0.15
            + min(context_row["children_ratio"] * 2.5, 1.0) * 0.10
        ),
        "Health Support": (
            (1.0 if has_disabled_member else 0.0) * 0.45
            + (1.0 if has_chronic_illness else 0.0) * 0.25
            + healthcare_access_inverse * 0.20
            + min(context_row["elderly_ratio"] * 2.0, 1.0) * 0.10
        ),
        "Elderly Care": (
            (1.0 if has_elderly else 0.0) * 0.55
            + min(context_row["elderly_ratio"] * 2.5, 1.0) * 0.25
            + poverty_score * 0.10
            + healthcare_access_inverse * 0.10
        ),
        "Disaster Relief": (
            (1.0 if recent_disaster_impact else 0.0) * 0.60
            + min(max(context_row["persons_per_housing_unit"] - 3, 0) / 3, 1.0) * 0.15
            + poverty_score * 0.15
            + min(context_row["children_ratio"] * 2.0, 1.0) * 0.10
        ),
    }

    vulnerability_score = (
        poverty_score * 0.30
        + food_insecurity_norm * 0.20
        + healthcare_access_inverse * 0.15
        + (1.0 if has_disabled_member else 0.0) * 0.15
        + (1.0 if has_elderly else 0.0) * 0.08
        + (1.0 if recent_disaster_impact else 0.0) * 0.07
        + min(max(context_row["persons_per_housing_unit"] - 3, 0) / 3, 1.0) * 0.05
    )

    if vulnerability_score >= 0.67:
        priority_level = "High"
    elif vulnerability_score >= 0.34:
        priority_level = "Medium"
    else:
        priority_level = "Low"

    sorted_services = sorted(service_scores.items(), key=lambda item: item[1], reverse=True)
    recommended_services = [
        {"service": service_name, "score": round(float(score), 3)}
        for service_name, score in sorted_services
        if score >= MIN_SERVICE_CONFIDENCE
    ][:3]
    recommended_services = apply_service_guardrails(payload, recommended_services)

    matched_programs = match_programs_to_assessment(payload, recommended_services)
    direct_welfare_programs, referral_programs = split_program_matches(matched_programs)
    outcome_type = determine_outcome_type(recommended_services, direct_welfare_programs, referral_programs, priority_level)
    household_reasons, location_reasons = build_reason_breakdown(
        payload,
        context_row,
        recommended_services,
        matched_programs,
        priority_level,
    )

    return {
        "recommended_services": recommended_services,
        "priority_level": priority_level,
        "priority_score": round(float(vulnerability_score), 3),
        "outcome_type": outcome_type,
        "result_summary": build_outcome_summary(payload, recommended_services, direct_welfare_programs, referral_programs),
        "household_reasons": household_reasons,
        "location_reasons": location_reasons,
        "top_factors": (household_reasons + location_reasons)[:4],
        "matched_programs": matched_programs,
        "direct_welfare_programs": direct_welfare_programs,
        "referral_programs": referral_programs,
        "next_steps": build_next_steps(recommended_services, matched_programs),
        "location_context": {
            "district_name": context_row["display_district_name"],
            "ds_division_name": context_row["display_ds_division_name"],
            "gn_division_name": context_row["display_gn_division_name"],
            "population_total": int(context_row["population_total"]),
            "occupied_housing_units": int(round(context_row["occupied_housing_units"])),
            "elderly_ratio": round(float(context_row["elderly_ratio"]), 3),
            "children_ratio": round(float(context_row["children_ratio"]), 3),
        },
        "assessment_method": "gn_context_scoring_v1",
    }


def predict_citizen_assessment(payload: dict[str, Any], context_row: pd.Series) -> dict[str, Any]:
    if not citizen_service_model or not citizen_priority_model or not citizen_model_metadata:
        return scoring_fallback_assessment(payload, context_row)

    feature_row = build_citizen_feature_row(payload, context_row)
    feature_df = pd.DataFrame([feature_row])

    service_probabilities = citizen_service_model.predict_proba(feature_df)
    service_labels = citizen_model_metadata["service_labels"]
    service_name_map = citizen_model_metadata["service_name_map"]

    recommended_services = []
    for idx, label in enumerate(service_labels):
        estimator_probs = service_probabilities[idx][0]
        positive_probability = float(estimator_probs[-1])
        if positive_probability >= get_service_threshold(label):
            recommended_services.append(
                {"service": service_name_map[label], "score": round(positive_probability, 3)}
            )

    recommended_services.sort(key=lambda item: item["score"], reverse=True)
    recommended_services = apply_service_guardrails(payload, recommended_services[:3])

    no_strong_match_probability = None
    if citizen_outcome_model is not None and citizen_model_metadata:
        no_strong_match_probability = float(citizen_outcome_model.predict_proba(feature_df)[0][-1])
        no_strong_match_threshold = float(citizen_model_metadata.get("no_strong_match_threshold", 0.5))
        if no_strong_match_probability >= no_strong_match_threshold:
            recommended_services = []

    classifier = citizen_priority_model.named_steps.get("classifier")
    priority_prediction = citizen_priority_model.predict(feature_df)[0]
    priority_probabilities = citizen_priority_model.predict_proba(feature_df)[0]
    priority_classes = list(getattr(classifier, "classes_", []))
    if priority_classes:
        probability_map = {
            str(priority_classes[idx]): float(priority_probabilities[idx])
            for idx in range(min(len(priority_classes), len(priority_probabilities)))
        }
        priority_score = round(probability_map.get(priority_prediction, float(np.max(priority_probabilities))), 3)
    else:
        priority_score = round(float(np.max(priority_probabilities)), 3)

    matched_programs = match_programs_to_assessment(payload, recommended_services)
    direct_welfare_programs, referral_programs = split_program_matches(matched_programs)
    outcome_type = determine_outcome_type(recommended_services, direct_welfare_programs, referral_programs, priority_prediction)
    household_reasons, location_reasons = build_reason_breakdown(
        payload,
        context_row,
        recommended_services,
        matched_programs,
        priority_prediction,
    )

    return {
        "recommended_services": recommended_services,
        "priority_level": priority_prediction,
        "priority_score": priority_score,
        "no_strong_match_probability": round(no_strong_match_probability, 3) if no_strong_match_probability is not None else None,
        "outcome_type": outcome_type,
        "result_summary": build_outcome_summary(payload, recommended_services, direct_welfare_programs, referral_programs),
        "household_reasons": household_reasons,
        "location_reasons": location_reasons,
        "top_factors": (household_reasons + location_reasons)[:4],
        "matched_programs": matched_programs,
        "direct_welfare_programs": direct_welfare_programs,
        "referral_programs": referral_programs,
        "next_steps": build_next_steps(recommended_services, matched_programs),
        "location_context": {
            "district_name": context_row["display_district_name"],
            "ds_division_name": context_row["display_ds_division_name"],
            "gn_division_name": context_row["display_gn_division_name"],
            "population_total": int(context_row["population_total"]),
            "occupied_housing_units": int(round(context_row["occupied_housing_units"])),
            "elderly_ratio": round(float(context_row["elderly_ratio"]), 3),
            "children_ratio": round(float(context_row["children_ratio"]), 3),
        },
        "assessment_method": "citizen_guidance_ml_v1",
    }


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "ok",
            "message": "Backend is running",
            "citizen_ml_models_loaded": bool(citizen_service_model and citizen_priority_model and citizen_model_metadata),
            "citizen_outcome_model_loaded": bool(citizen_outcome_model),
        }
    )


@app.route("/api/locations", methods=["GET"])
def get_locations():
    return jsonify(get_location_tree())


@app.route("/api/program-catalog", methods=["GET"])
def get_program_catalog():
    return jsonify(program_catalog or {"metadata": {}, "programs": []})


@app.route("/api/assess-citizen", methods=["POST"])
def assess_citizen():
    try:
        payload = request.get_json() or {}
        missing = [field for field in LOCATION_REQUIRED_FIELDS if field not in payload or str(payload[field]).strip() == ""]
        if missing:
            return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

        validation_errors = validate_assessment_payload(payload)
        if validation_errors:
            return jsonify({"error": " ".join(validation_errors)}), 400

        context_row = get_gn_context(
            payload["district_name"], payload["ds_division_name"], payload["gn_division_name"]
        )
        if context_row is None:
            return jsonify({"error": "Selected GN division was not found in the official context dataset."}), 404

        result = predict_citizen_assessment(payload, context_row)
        return jsonify({"success": True, **result})
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
