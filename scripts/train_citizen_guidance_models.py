from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


SERVICE_LABELS = [
    "food_assistance_label",
    "health_support_label",
    "elderly_care_label",
    "disaster_relief_label",
]

SERVICE_NAME_MAP = {
    "food_assistance_label": "Food Assistance",
    "health_support_label": "Health Support",
    "elderly_care_label": "Elderly Care",
    "disaster_relief_label": "Disaster Relief",
}

FEATURE_COLUMNS = [
    "district_code",
    "ds_division_code",
    "population_total",
    "occupied_housing_units",
    "children_ratio",
    "elderly_ratio",
    "persons_per_housing_unit",
    "housing_units_log",
    "population_log",
    "family_size",
    "income_value",
    "employment_status",
    "has_elderly",
    "has_disabled_member",
    "has_chronic_illness",
    "recent_disaster_impact",
    "food_insecurity_score",
    "healthcare_access_score",
]

CATEGORICAL_FEATURES = ["employment_status"]
NUMERIC_FEATURES = [feature for feature in FEATURE_COLUMNS if feature not in CATEGORICAL_FEATURES]
PRIORITY_CLASSES = ["Low", "Medium", "High"]

INCOME_RANGE_MIDPOINTS = {
    "below-25000": 15000,
    "25000-50000": 37500,
    "50000-100000": 75000,
    "100000-200000": 150000,
    "above-200000": 250000,
}


def choose_income_range(rng: np.random.Generator, poverty_signal: float) -> tuple[str, int]:
    if np.isnan(poverty_signal):
        poverty_signal = 0.5
    probabilities = np.array(
        [
            0.34 + poverty_signal * 0.20,
            0.28 + poverty_signal * 0.08,
            0.22 - poverty_signal * 0.12,
            0.11 - poverty_signal * 0.09,
            0.05 - poverty_signal * 0.07,
        ]
    )
    probabilities = np.clip(probabilities, 0.02, None)
    probabilities = probabilities / probabilities.sum()
    labels = list(INCOME_RANGE_MIDPOINTS.keys())
    selected = rng.choice(labels, p=probabilities)
    return selected, INCOME_RANGE_MIDPOINTS[selected]


def employment_distribution(elderly_ratio: float, poverty_signal: float) -> tuple[list[str], list[float]]:
    if np.isnan(elderly_ratio):
        elderly_ratio = 0.0
    if np.isnan(poverty_signal):
        poverty_signal = 0.5
    options = ["employed", "self-employed", "unemployed", "retired", "student", "unable-to-work"]
    weights = np.array(
        [
            0.36 - poverty_signal * 0.10,
            0.18,
            0.20 + poverty_signal * 0.10,
            0.08 + elderly_ratio * 0.18,
            0.10,
            0.08 + elderly_ratio * 0.08 + poverty_signal * 0.05,
        ]
    )
    weights = np.clip(weights, 0.03, None)
    weights = weights / weights.sum()
    return options, weights.tolist()


def build_priority_and_service_targets(sample: dict[str, float | int | str], context: pd.Series) -> tuple[dict[str, int], str, float]:
    family_size = max(int(sample["family_size"]), 1)
    income_value = float(sample["income_value"])
    income_per_capita = income_value / family_size
    food_insecurity_score = int(sample["food_insecurity_score"])
    healthcare_access_score = int(sample["healthcare_access_score"])
    has_elderly = int(sample["has_elderly"]) == 1
    has_disabled_member = int(sample["has_disabled_member"]) == 1
    has_chronic_illness = int(sample["has_chronic_illness"]) == 1
    recent_disaster_impact = int(sample["recent_disaster_impact"]) == 1
    employment_status = str(sample["employment_status"])

    poverty_score = max(0.0, min(1.0, 1 - min(income_per_capita / 50000, 1)))
    employment_risk = (
        1.0
        if employment_status in {"unemployed", "unable-to-work"}
        else 0.4
        if employment_status == "retired"
        else 0.2
    )
    healthcare_access_inverse = 1 - ((healthcare_access_score - 1) / 4)
    food_insecurity_norm = (food_insecurity_score - 1) / 4

    service_scores = {
        "food_assistance_label": (
            poverty_score * 0.45
            + food_insecurity_norm * 0.30
            + employment_risk * 0.15
            + min(float(context["children_ratio"]) * 2.5, 1.0) * 0.10
        ),
        "health_support_label": (
            (1.0 if has_disabled_member else 0.0) * 0.45
            + (1.0 if has_chronic_illness else 0.0) * 0.25
            + healthcare_access_inverse * 0.20
            + min(float(context["elderly_ratio"]) * 2.0, 1.0) * 0.10
        ),
        "elderly_care_label": (
            (1.0 if has_elderly else 0.0) * 0.55
            + min(float(context["elderly_ratio"]) * 2.5, 1.0) * 0.25
            + poverty_score * 0.10
            + healthcare_access_inverse * 0.10
        ),
        "disaster_relief_label": (
            (1.0 if recent_disaster_impact else 0.0) * 0.60
            + min(max(float(context["persons_per_housing_unit"]) - 3, 0) / 3, 1.0) * 0.15
            + poverty_score * 0.15
            + min(float(context["children_ratio"]) * 2.0, 1.0) * 0.10
        ),
    }

    labels = {key: int(score >= 0.35) for key, score in service_scores.items()}

    vulnerability_score = (
        poverty_score * 0.30
        + food_insecurity_norm * 0.20
        + healthcare_access_inverse * 0.15
        + (1.0 if has_disabled_member else 0.0) * 0.15
        + (1.0 if has_elderly else 0.0) * 0.08
        + (1.0 if recent_disaster_impact else 0.0) * 0.07
        + min(max(float(context["persons_per_housing_unit"]) - 3, 0) / 3, 1.0) * 0.05
    )

    if vulnerability_score >= 0.67:
        priority = "High"
    elif vulnerability_score >= 0.34:
        priority = "Medium"
    else:
        priority = "Low"

    return labels, priority, round(float(vulnerability_score), 3)


def synthesize_training_dataset(context_df: pd.DataFrame, samples_per_gn: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict[str, float | int | str]] = []

    for _, context in context_df.iterrows():
        persons_per_housing_unit = float(context["persons_per_housing_unit"]) if pd.notna(context["persons_per_housing_unit"]) else 3.5
        elderly_ratio = float(context["elderly_ratio"]) if pd.notna(context["elderly_ratio"]) else 0.1
        children_ratio = float(context["children_ratio"]) if pd.notna(context["children_ratio"]) else 0.2

        poverty_signal = np.clip(((persons_per_housing_unit - 3.1) / 2.2) + children_ratio * 0.6, 0, 1)
        base_family_size = int(np.clip(np.round(persons_per_housing_unit + rng.normal(0, 0.7)), 1, 8))

        employment_options, employment_probs = employment_distribution(elderly_ratio, poverty_signal)

        for _ in range(samples_per_gn):
            family_size = int(np.clip(base_family_size + rng.integers(-1, 2), 1, 10))
            income_range, income_value = choose_income_range(rng, poverty_signal)

            has_elderly = int(rng.random() < min(0.12 + elderly_ratio * 2.6, 0.95))
            has_disabled_member = int(rng.random() < min(0.06 + elderly_ratio * 0.35 + poverty_signal * 0.18, 0.65))
            has_chronic_illness = int(
                rng.random() < min(0.08 + has_elderly * 0.24 + has_disabled_member * 0.20 + elderly_ratio * 0.18, 0.82)
            )
            recent_disaster_impact = int(rng.random() < min(0.03 + poverty_signal * 0.07, 0.25))

            food_insecurity_score = int(
                np.clip(np.round(1 + poverty_signal * 3 + rng.normal(0, 0.8)), 1, 5)
            )
            healthcare_access_score = int(
                np.clip(np.round(4 - poverty_signal * 1.4 - has_chronic_illness * 0.3 + rng.normal(0, 0.7)), 1, 5)
            )

            sample = {
                "district_code": int(context["district_code"]),
                "ds_division_code": int(context["ds_division_code"]),
                "population_total": float(context["population_total"]),
                "occupied_housing_units": float(context["occupied_housing_units"]),
                "children_ratio": children_ratio,
                "elderly_ratio": elderly_ratio,
                "persons_per_housing_unit": persons_per_housing_unit,
                "housing_units_log": float(context["housing_units_log"]),
                "population_log": float(context["population_log"]),
                "family_size": family_size,
                "income_value": income_value,
                "employment_status": rng.choice(employment_options, p=employment_probs),
                "has_elderly": has_elderly,
                "has_disabled_member": has_disabled_member,
                "has_chronic_illness": has_chronic_illness,
                "recent_disaster_impact": recent_disaster_impact,
                "food_insecurity_score": food_insecurity_score,
                "healthcare_access_score": healthcare_access_score,
                "district_name": context["district_name"],
                "ds_division_name": context["ds_division_name"],
                "gn_division_name": context["gn_division_name"],
                "income_range": income_range,
            }

            labels, priority, vulnerability_score = build_priority_and_service_targets(sample, context)
            sample.update(labels)
            sample["priority_level"] = priority
            sample["vulnerability_score"] = vulnerability_score
            records.append(sample)

    return pd.DataFrame.from_records(records)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )


def train_models(training_df: pd.DataFrame, model_dir: Path) -> dict[str, object]:
    X = training_df[FEATURE_COLUMNS]
    y_services = training_df[SERVICE_LABELS]
    y_priority = training_df["priority_level"]

    X_train, X_test, y_services_train, y_services_test, y_priority_train, y_priority_test = train_test_split(
        X,
        y_services,
        y_priority,
        test_size=0.2,
        random_state=42,
        stratify=y_priority,
    )

    service_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                MultiOutputClassifier(
                    RandomForestClassifier(
                        n_estimators=180,
                        max_depth=14,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    )
                ),
            ),
        ]
    )

    priority_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=220,
                    max_depth=16,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )

    service_model.fit(X_train, y_services_train)
    priority_model.fit(X_train, y_priority_train)

    service_predictions = service_model.predict(X_test)
    priority_predictions = priority_model.predict(X_test)

    service_metrics = {}
    for idx, label in enumerate(SERVICE_LABELS):
        service_metrics[label] = {
            "f1": round(float(f1_score(y_services_test[label], service_predictions[:, idx], zero_division=0)), 4),
            "positive_rate": round(float(y_services_test[label].mean()), 4),
        }

    priority_metrics = {
        "accuracy": round(float(accuracy_score(y_priority_test, priority_predictions)), 4),
        "macro_f1": round(float(f1_score(y_priority_test, priority_predictions, average="macro", labels=PRIORITY_CLASSES)), 4),
    }

    joblib.dump(service_model, model_dir / "citizen_service_model.pkl")
    joblib.dump(priority_model, model_dir / "citizen_priority_model.pkl")

    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "service_labels": SERVICE_LABELS,
        "service_name_map": SERVICE_NAME_MAP,
        "priority_classes": PRIORITY_CLASSES,
        "service_metrics": service_metrics,
        "priority_metrics": priority_metrics,
    }
    (model_dir / "citizen_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train citizen welfare guidance models.")
    parser.add_argument(
        "--context-csv",
        type=Path,
        default=Path("data/research/gn_official_context_dataset.csv"),
        help="Path to the merged GN official context dataset.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=Path("data/research/citizen_guidance_training_dataset.csv"),
        help="Path to write the synthesized citizen training dataset.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("backend/models"),
        help="Directory where trained models will be written.",
    )
    parser.add_argument("--samples-per-gn", type=int, default=5, help="Synthetic samples to create per GN division.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    context_df = pd.read_csv(args.context_csv)
    training_df = synthesize_training_dataset(context_df, args.samples_per_gn, args.seed)

    args.output_dataset.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(args.output_dataset, index=False)

    args.model_dir.mkdir(parents=True, exist_ok=True)
    metadata = train_models(training_df, args.model_dir)

    print(f"training_rows={len(training_df)}")
    print(f"training_columns={len(training_df.columns)}")
    print("service_metrics=", json.dumps(metadata["service_metrics"], indent=2))
    print("priority_metrics=", json.dumps(metadata["priority_metrics"], indent=2))
    print(f"saved_dataset={args.output_dataset}")
    print(f"saved_models={args.model_dir}")


if __name__ == "__main__":
    main()
