from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app as backend_app


def make_context_row(**overrides):
    context = {
        "district_code": 1,
        "ds_division_code": 1,
        "population_total": 1448.0,
        "occupied_housing_units": 400.0,
        "children_ratio": 0.18,
        "elderly_ratio": 0.183,
        "persons_per_housing_unit": 3.2,
        "housing_units_log": 5.99,
        "population_log": 7.28,
        "display_district_name": "Badulla",
        "display_ds_division_name": "Ella",
        "display_gn_division_name": "Madhuragama",
    }
    context.update(overrides)
    return pd.Series(context)


def test_service_linked_program_requires_trigger_match():
    payload = {
        "district_name": "badulla",
        "ds_division_name": "ella",
        "gn_division_name": "madhuragama",
        "family_size": 1,
        "income_range": "above-200000",
        "employment_status": "employed",
        "has_elderly": 0,
        "has_disabled_member": 0,
        "has_chronic_illness": 0,
        "recent_disaster_impact": 0,
        "food_insecurity_score": 1,
        "healthcare_access_score": 5,
    }

    matched_programs = backend_app.match_programs_to_assessment(
        payload,
        [{"service": "Food Assistance", "score": 0.52}],
    )

    assert all(program["id"] != "aswesuma" for program in matched_programs)


def test_low_signal_household_has_no_forced_service_recommendation(monkeypatch):
    monkeypatch.setattr(backend_app, "citizen_service_model", None)
    monkeypatch.setattr(backend_app, "citizen_priority_model", None)
    monkeypatch.setattr(backend_app, "citizen_model_metadata", None)
    monkeypatch.setattr(backend_app, "get_gn_context", lambda *_args: make_context_row())

    payload = {
        "district_name": "Badulla",
        "ds_division_name": "Ella",
        "gn_division_name": "Madhuragama",
        "family_size": 1,
        "income_range": "above-200000",
        "employment_status": "employed",
        "has_elderly": 0,
        "has_disabled_member": 0,
        "has_chronic_illness": 0,
        "recent_disaster_impact": 0,
        "food_insecurity_score": 1,
        "healthcare_access_score": 5,
    }

    client = backend_app.app.test_client()
    response = client.post("/api/assess-citizen", json=payload)

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["recommended_services"] == []
    assert data["matched_programs"] == []
    assert data["outcome_type"] == "no_strong_match"
    assert data["priority_level"] == "Low"
    assert "Your household inputs do not indicate a strong direct welfare need at this time." in data["household_reasons"]
    assert "No service or programme trigger was strongly matched from the current household details." in data["household_reasons"]
    assert "Location context is shown below for reference, but area-level indicators alone did not create a direct programme match." in data["location_reasons"]
    assert "The selected GN division has a relatively high elderly population share" not in data["household_reasons"]


def test_trigger_matched_household_still_receives_program_guidance(monkeypatch):
    monkeypatch.setattr(backend_app, "citizen_service_model", None)
    monkeypatch.setattr(backend_app, "citizen_priority_model", None)
    monkeypatch.setattr(backend_app, "citizen_model_metadata", None)
    monkeypatch.setattr(
        backend_app,
        "get_gn_context",
        lambda *_args: make_context_row(children_ratio=0.32, elderly_ratio=0.08, persons_per_housing_unit=4.6),
    )

    payload = {
        "district_name": "Badulla",
        "ds_division_name": "Ella",
        "gn_division_name": "Madhuragama",
        "family_size": 4,
        "income_range": "below-25000",
        "employment_status": "unemployed",
        "has_elderly": 0,
        "has_disabled_member": 0,
        "has_chronic_illness": 0,
        "recent_disaster_impact": 0,
        "food_insecurity_score": 4,
        "healthcare_access_score": 3,
    }

    client = backend_app.app.test_client()
    response = client.post("/api/assess-citizen", json=payload)

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert any(service["service"] == "Food Assistance" for service in data["recommended_services"])
    assert any(program["id"] == "aswesuma" for program in data["matched_programs"])
    assert any(program["id"] == "aswesuma" for program in data["direct_welfare_programs"])
    assert data["outcome_type"] == "direct_welfare_match"
    assert "meaningful welfare support match" in data["result_summary"].lower()


def test_household_elderly_factor_only_appears_when_household_has_elderly():
    payload = {
        "family_size": 1,
        "income_range": "above-200000",
        "has_elderly": 0,
        "has_disabled_member": 0,
        "has_chronic_illness": 0,
        "recent_disaster_impact": 0,
        "food_insecurity_score": 1,
    }

    household_reasons, location_reasons = backend_app.build_reason_breakdown(payload, make_context_row(elderly_ratio=0.201))

    assert "The household includes an elderly family member who may need support" not in household_reasons
    assert "Your household inputs do not indicate a strong direct welfare need at this time." in household_reasons
    assert "Location context is shown below for reference, but area-level indicators alone did not create a direct programme match." in location_reasons


def test_referral_case_explains_household_trigger_before_location_context():
    payload = {
        "family_size": 1,
        "income_range": "above-200000",
        "employment_status": "self-employed",
        "has_elderly": 0,
        "has_disabled_member": 0,
        "has_chronic_illness": 0,
        "recent_disaster_impact": 0,
        "food_insecurity_score": 1,
    }

    household_reasons, location_reasons = backend_app.build_reason_breakdown(
        payload,
        make_context_row(elderly_ratio=0.201),
        recommended_services=[],
        matched_programs=[{"id": "sarvodaya_finance_referral"}],
        priority_level="Low",
    )

    assert "Self-employment status may match livelihood or finance referral pathways" in household_reasons
    assert "The selected GN division has a relatively high elderly population share" not in household_reasons
    assert location_reasons == []


def test_referral_only_case_is_split_out_from_direct_welfare():
    payload = {
        "district_name": "Badulla",
        "ds_division_name": "Ella",
        "gn_division_name": "Madhuragama",
        "family_size": 1,
        "income_range": "above-200000",
        "employment_status": "self-employed",
        "has_elderly": 0,
        "has_disabled_member": 0,
        "has_chronic_illness": 0,
        "recent_disaster_impact": 0,
        "food_insecurity_score": 1,
        "healthcare_access_score": 5,
    }

    recommended_services = []
    matched_programs = backend_app.match_programs_to_assessment(payload, recommended_services)
    direct_welfare_programs, referral_programs = backend_app.split_program_matches(matched_programs)
    result_summary = backend_app.build_outcome_summary(payload, recommended_services, direct_welfare_programs, referral_programs)
    outcome_type = backend_app.determine_outcome_type(recommended_services, direct_welfare_programs, referral_programs, "Low")

    assert direct_welfare_programs == []
    assert any(program["id"] == "sarvodaya_finance_referral" for program in referral_programs)
    assert outcome_type == "referral_only"
    assert "self-employment status matched a livelihood or finance referral" in result_summary.lower()


def test_elderly_care_service_is_blocked_without_household_elderly():
    payload = {
        "district_name": "Kandy",
        "ds_division_name": "Udadumbara",
        "gn_division_name": "Hanwella",
        "family_size": 1,
        "income_range": "above-200000",
        "employment_status": "employed",
        "has_elderly": 0,
        "has_disabled_member": 0,
        "has_chronic_illness": 0,
        "recent_disaster_impact": 0,
        "food_insecurity_score": 3,
        "healthcare_access_score": 3,
    }

    filtered = backend_app.apply_service_guardrails(
        payload,
        [{"service": "Elderly Care", "score": 0.95}],
    )

    assert filtered == []


def test_invalid_payload_is_rejected(monkeypatch):
    monkeypatch.setattr(backend_app, "get_gn_context", lambda *_args: make_context_row())

    payload = {
        "district_name": "Badulla",
        "ds_division_name": "Ella",
        "gn_division_name": "Madhuragama",
        "family_size": 0,
        "income_range": "not-valid",
        "employment_status": "invalid",
        "has_elderly": 3,
        "has_disabled_member": 0,
        "has_chronic_illness": 0,
        "recent_disaster_impact": 0,
        "food_insecurity_score": 7,
        "healthcare_access_score": 0,
    }

    client = backend_app.app.test_client()
    response = client.post("/api/assess-citizen", json=payload)

    assert response.status_code == 400
    data = response.get_json()
    assert "Family size must be between 1 and 20." in data["error"]
    assert "Monthly income range is invalid." in data["error"]
