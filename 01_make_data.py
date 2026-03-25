# ============================================================
# 01_make_data.py
# Synthetic pet health data generation and CSV export
# Copyright 2026 YIRANG JUNG
# Licensed under the Apache License, Version 2.0
# ============================================================

import os
import numpy as np
import pandas as pd


def create_synthetic_pet_health_data(n_samples=1200, random_state=42):
    """Generate synthetic tabular pet health data."""
    rng = np.random.default_rng(random_state)

    age = rng.integers(1, 18, size=n_samples)

    weight = rng.normal(7.0, 2.5, size=n_samples)
    weight = np.clip(weight, 1.5, 20.0)

    temperature = rng.normal(38.7, 0.5, size=n_samples)
    temperature = np.clip(temperature, 37.5, 40.5)

    heart_rate = rng.normal(120, 20, size=n_samples)
    heart_rate = np.clip(heart_rate, 70, 200)

    appetite_score = rng.integers(1, 11, size=n_samples)
    activity_score = rng.integers(1, 11, size=n_samples)

    wbc = rng.normal(11.0, 3.0, size=n_samples)
    wbc = np.clip(wbc, 4.0, 25.0)

    rbc = rng.normal(6.0, 0.8, size=n_samples)
    rbc = np.clip(rbc, 3.5, 8.5)

    glucose = rng.normal(105, 20, size=n_samples)
    glucose = np.clip(glucose, 60, 220)

    risk_score = (
        0.18 * age
        + 1.8 * (temperature - 38.5)
        + 0.03 * (heart_rate - 120)
        - 0.45 * appetite_score
        - 0.40 * activity_score
        + 0.22 * (wbc - 11)
        - 0.55 * (rbc - 6)
        + 0.025 * (glucose - 100)
        + rng.normal(0, 0.8, size=n_samples)
    )

    risk_level = np.where(
        risk_score < 0.5,
        "normal",
        np.where(risk_score < 2.5, "caution", "danger")
    )

    df = pd.DataFrame({
        "age": age,
        "weight": np.round(weight, 2),
        "temperature": np.round(temperature, 2),
        "heart_rate": np.round(heart_rate, 0).astype(int),
        "appetite_score": appetite_score,
        "activity_score": activity_score,
        "wbc": np.round(wbc, 2),
        "rbc": np.round(rbc, 2),
        "glucose": np.round(glucose, 2),
        "risk_level": risk_level,
    })
    return df


def main():
    base_dir = os.path.abspath(".")
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, "pet_health_data.csv")
    df = create_synthetic_pet_health_data(n_samples=1200, random_state=42)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("Synthetic data generation complete")
    print(f"Saved to: {csv_path}")
    print("=" * 60)
    print("\nFirst 5 rows")
    print(df.head())
    print("\nClass distribution")
    print(df["risk_level"].value_counts())


if __name__ == "__main__":
    main()
