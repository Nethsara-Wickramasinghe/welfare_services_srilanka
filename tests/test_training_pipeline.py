from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_citizen_guidance_models.py"
spec = importlib.util.spec_from_file_location("train_citizen_guidance_models", SCRIPT_PATH)
training_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(training_module)


def make_context(**overrides):
    context = {
        "children_ratio": 0.18,
        "elderly_ratio": 0.22,
        "persons_per_housing_unit": 3.1,
    }
    context.update(overrides)
    return pd.Series(context)


def test_no_strong_match_label_is_explicit_for_low_need_case():
    sample = {
        "family_size": 1,
        "income_value": 250000,
        "employment_status": "employed",
        "has_elderly": 0,
        "has_disabled_member": 0,
        "has_chronic_illness": 0,
        "recent_disaster_impact": 0,
        "food_insecurity_score": 1,
        "healthcare_access_score": 5,
    }

    labels, priority, _ = training_module.build_priority_and_service_targets(sample, make_context())

    assert labels[training_module.NO_STRONG_MATCH_LABEL] == 1
    assert all(labels[label] == 0 for label in training_module.SERVICE_LABELS)
    assert priority == "Low"


def test_elderly_care_label_requires_household_elderly():
    sample = {
        "family_size": 1,
        "income_value": 250000,
        "employment_status": "employed",
        "has_elderly": 0,
        "has_disabled_member": 0,
        "has_chronic_illness": 0,
        "recent_disaster_impact": 0,
        "food_insecurity_score": 3,
        "healthcare_access_score": 3,
    }

    labels, _, _ = training_module.build_priority_and_service_targets(sample, make_context(elderly_ratio=0.35))

    assert labels["elderly_care_label"] == 0

