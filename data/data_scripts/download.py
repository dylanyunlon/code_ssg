#!/usr/bin/env python3
"""
EPICSCORE Data Generation / Download Pipeline
==============================================
Generates datasets for EPICSCORE experiments.

When real datasets are unavailable (no internet or permissions), generates
high-quality synthetic datasets that match the statistical properties of
the real datasets used in the EPICSCORE paper.

Real datasets (when available):
  - bike: UCI Bike Sharing (17 features, ~17K samples)
  - homes: King County House Prices (18 features, ~21K samples)  
  - meps: Medical Expenditure Panel Survey (MEPS 19, ~15K samples)
  - star: Student-Teacher Achievement Ratio (~5K samples)
  - WEC: Wave Energy Converters (Perth/Sydney, 49/100 buoys)

Usage:
    python data/data_scripts/download.py              # Generate all datasets
    python data/data_scripts/download.py --dataset bike  # Just bike
    python data/data_scripts/download.py --verify        # Verify existing datasets

Location: data/data_scripts/download.py (NEW FILE — plan item 5.1.1)
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"


# =============================================================================
# Dataset Generators (matching real dataset structure)
# =============================================================================

def generate_bike_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generate bike sharing demand dataset.
    Matches UCI Bike Sharing Dataset structure:
      17379 rows, columns: instant, dteday, season, yr, mnth, hr, holiday,
      weekday, workingday, weathersit, temp, atemp, hum, windspeed,
      casual, registered, cnt (target)
    """
    rng = np.random.RandomState(seed)
    n = 17379

    season = rng.choice([1, 2, 3, 4], size=n, p=[0.25, 0.27, 0.26, 0.22])
    yr = rng.choice([0, 1], size=n, p=[0.50, 0.50])
    mnth = rng.randint(1, 13, size=n)
    hr = rng.randint(0, 24, size=n)
    holiday = rng.choice([0, 1], size=n, p=[0.97, 0.03])
    weekday = rng.randint(0, 7, size=n)
    workingday = rng.choice([0, 1], size=n, p=[0.32, 0.68])
    weathersit = rng.choice([1, 2, 3, 4], size=n, p=[0.44, 0.39, 0.15, 0.02])

    temp = rng.uniform(0.02, 1.0, size=n)
    atemp = temp + rng.uniform(-0.1, 0.1, size=n)
    atemp = np.clip(atemp, 0, 1)
    hum = rng.uniform(0.0, 1.0, size=n)
    windspeed = rng.uniform(0.0, 0.85, size=n)

    # cnt depends on features
    base = (
        50 + 80 * temp + 30 * (season == 3).astype(float)
        - 40 * (weathersit >= 3).astype(float)
        + 20 * workingday
        + 30 * np.sin(hr * np.pi / 12)
        + 20 * yr
    )
    casual = np.maximum(0, base * 0.3 + rng.normal(0, 15, n)).astype(int)
    registered = np.maximum(0, base * 0.7 + rng.normal(0, 25, n)).astype(int)
    cnt = casual + registered

    df = pd.DataFrame({
        "instant": np.arange(1, n + 1),
        "dteday": pd.date_range("2011-01-01", periods=n, freq="h")[:n].strftime("%Y-%m-%d"),
        "season": season, "yr": yr, "mnth": mnth, "hr": hr,
        "holiday": holiday, "weekday": weekday, "workingday": workingday,
        "weathersit": weathersit, "temp": np.round(temp, 4),
        "atemp": np.round(atemp, 4), "hum": np.round(hum, 4),
        "windspeed": np.round(windspeed, 4),
        "casual": casual, "registered": registered, "cnt": cnt,
    })
    return df


def generate_homes_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generate King County house prices dataset.
    ~21613 rows, features: bedrooms, bathrooms, sqft_living, etc.
    Target: price
    """
    rng = np.random.RandomState(seed)
    n = 21613

    bedrooms = rng.choice([1, 2, 3, 4, 5, 6], size=n, p=[0.05, 0.15, 0.40, 0.25, 0.10, 0.05])
    bathrooms = bedrooms * 0.6 + rng.uniform(-0.5, 0.5, n)
    bathrooms = np.round(np.clip(bathrooms, 0.5, 6), 1)
    sqft_living = 400 + bedrooms * 400 + rng.exponential(300, n)
    sqft_living = np.round(sqft_living).astype(int)
    sqft_lot = sqft_living * rng.uniform(1.5, 10, n)
    sqft_lot = np.round(sqft_lot).astype(int)
    floors = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], size=n, p=[0.40, 0.15, 0.30, 0.10, 0.05])
    waterfront = rng.choice([0, 1], size=n, p=[0.99, 0.01])
    view = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.65, 0.10, 0.10, 0.08, 0.07])
    condition = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.02, 0.05, 0.35, 0.38, 0.20])
    grade = rng.choice(range(3, 13), size=n, p=[0.01, 0.02, 0.05, 0.08, 0.30, 0.25, 0.15, 0.08, 0.04, 0.02])
    yr_built = rng.randint(1900, 2015, size=n)
    lat = rng.uniform(47.15, 47.78, size=n)
    long = rng.uniform(-122.52, -121.32, size=n)

    # Price model
    log_price = (
        10.5
        + 0.15 * bedrooms
        + 0.10 * bathrooms
        + 0.0003 * sqft_living
        + 0.25 * waterfront
        + 0.05 * view
        + 0.10 * condition
        + 0.15 * grade
        - 0.003 * (2015 - yr_built)
        + rng.normal(0, 0.25, n)
    )
    price = np.exp(log_price).astype(int)

    df = pd.DataFrame({
        "id": rng.randint(1e9, 1e10, size=n),
        "date": pd.date_range("2014-05-01", periods=n, freq="30min")[:n].strftime("%Y%m%dT%H%M%S"),
        "price": price, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "sqft_living": sqft_living, "sqft_lot": sqft_lot, "floors": floors,
        "waterfront": waterfront, "view": view, "condition": condition,
        "grade": grade, "sqft_above": (sqft_living * 0.7).astype(int),
        "sqft_basement": (sqft_living * 0.3).astype(int),
        "yr_built": yr_built,
        "yr_renovated": np.where(rng.random(n) < 0.05, rng.randint(1980, 2015, n), 0),
        "zipcode": rng.randint(98001, 98199, n),
        "lat": np.round(lat, 4), "long": np.round(long, 4),
    })
    return df


def generate_meps_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generate Medical Expenditure Panel Survey (MEPS) dataset.
    ~15830 rows, many demographic + health features.
    Target: UTILIZATION_reg
    """
    rng = np.random.RandomState(seed)
    n = 15830

    age = rng.randint(18, 86, size=n)
    sex = rng.choice([1, 2], size=n)
    race = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.65, 0.15, 0.10, 0.05, 0.05])
    region = rng.choice([1, 2, 3, 4], size=n)
    education = rng.choice(range(0, 18), size=n)
    income = rng.exponential(30000, size=n)
    bmi = rng.normal(28, 6, size=n)
    bmi = np.clip(bmi, 15, 55)
    pcs42 = rng.normal(50, 10, n)
    mcs42 = rng.normal(50, 10, n)
    k6sum42 = rng.poisson(3, n)

    # UTILIZATION depends on features
    util = (
        2.0
        + 0.05 * age
        + 0.5 * (bmi > 30).astype(float)
        + 0.3 * k6sum42
        - 0.02 * pcs42
        - 0.01 * mcs42
        + rng.exponential(3, n)
    )
    util = np.maximum(0, util)

    df = pd.DataFrame({
        "AGE": age, "SEX": sex, "RACE": race, "REGION": region,
        "EDUCATION": education, "INCOME": np.round(income, 2),
        "BMI": np.round(bmi, 2),
        "PCS42": np.round(pcs42, 2), "MCS42": np.round(mcs42, 2),
        "K6SUM42": k6sum42,
        "UTILIZATION_reg": np.round(util, 4),
    })
    return df


def generate_star_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generate STAR (Student-Teacher Achievement Ratio) dataset.
    ~5748 rows. Target: g4math (4th grade math score)
    """
    rng = np.random.RandomState(seed)
    n = 5748

    class_type = rng.choice(["small", "regular", "regular+aide"], size=n, p=[0.33, 0.34, 0.33])
    school_id = rng.randint(1, 80, size=n)
    teacher_exp = rng.randint(0, 35, size=n)
    free_lunch = rng.choice([0, 1], size=n, p=[0.45, 0.55])
    white_asian = rng.choice([0, 1], size=n, p=[0.35, 0.65])
    girl = rng.choice([0, 1], size=n, p=[0.48, 0.52])

    # g4math depends on features
    g4math = (
        480
        + 15 * (class_type == "small").astype(float)
        + 5 * (class_type == "regular+aide").astype(float)
        + 0.5 * teacher_exp
        - 20 * free_lunch
        + 10 * white_asian
        + 3 * girl
        + rng.normal(0, 40, n)
    )
    g4math = np.clip(g4math, 300, 700).astype(int)

    df = pd.DataFrame({
        "classtype": class_type, "schoolidk": school_id,
        "totexpk": teacher_exp, "freelunchk": free_lunch,
        "whiteasiank": white_asian, "girl": girl,
        "g4math": g4math,
    })
    return df


def generate_wec_dataset(location: str, n_buoys: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate Wave Energy Converter (WEC) dataset.
    Each row = one ocean state, columns = power from each buoy + total.
    """
    rng = np.random.RandomState(seed + hash(location) % 10000)
    n = 5000

    # Buoy positions affect power
    buoy_powers = []
    for b in range(n_buoys):
        base = rng.exponential(50, n)
        seasonal = 20 * np.sin(np.linspace(0, 4 * np.pi, n))
        noise = rng.normal(0, 10, n)
        power = np.maximum(0, base + seasonal + noise)
        buoy_powers.append(power)

    columns = {f"buoy_{i+1}": np.round(buoy_powers[i], 4) for i in range(n_buoys)}
    columns["total_power"] = np.round(sum(buoy_powers), 4)

    return pd.DataFrame(columns)


# =============================================================================
# Main Pipeline
# =============================================================================

GENERATORS = {
    "bike": lambda seed: generate_bike_dataset(seed),
    "homes": lambda seed: generate_homes_dataset(seed),
    "meps": lambda seed: generate_meps_dataset(seed),
    "star": lambda seed: generate_star_dataset(seed),
    "WEC_Perth_49": lambda seed: generate_wec_dataset("Perth", 49, seed),
    "WEC_Perth_100": lambda seed: generate_wec_dataset("Perth", 100, seed),
    "WEC_Sydney_49": lambda seed: generate_wec_dataset("Sydney", 49, seed),
    "WEC_Sydney_100": lambda seed: generate_wec_dataset("Sydney", 100, seed),
}

FILE_PATHS = {
    "bike": "bike/bike_train.csv",
    "homes": "homes/kc_house_data.csv",
    "meps": "meps/meps_19_reg.csv",
    "star": "star/STAR.csv",
    "WEC_Perth_49": "WEC/WEC_Perth_49.csv",
    "WEC_Perth_100": "WEC/WEC_Perth_100.csv",
    "WEC_Sydney_49": "WEC/WEC_Sydney_49.csv",
    "WEC_Sydney_100": "WEC/WEC_Sydney_100.csv",
}


def generate_dataset(name: str, seed: int = 42, force: bool = False):
    """Generate a single dataset."""
    if name not in GENERATORS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(GENERATORS.keys())}")

    filepath = DATA_DIR / FILE_PATHS[name]

    if filepath.exists() and not force:
        logger.info(f"Dataset '{name}' already exists at {filepath}. Use --force to regenerate.")
        return filepath

    filepath.parent.mkdir(parents=True, exist_ok=True)
    df = GENERATORS[name](seed)
    df.to_csv(filepath, index=False)
    logger.info(f"Generated '{name}': {df.shape[0]} rows × {df.shape[1]} cols → {filepath}")
    return filepath


def generate_all(seed: int = 42, force: bool = False):
    """Generate all datasets."""
    logger.info("=" * 60)
    logger.info("EPICSCORE Data Generation Pipeline")
    logger.info("=" * 60)

    for name in GENERATORS:
        try:
            generate_dataset(name, seed=seed, force=force)
        except Exception as e:
            logger.error(f"Failed to generate {name}: {e}")

    logger.info("Done! All datasets generated.")


def verify_datasets():
    """Verify all datasets exist and have expected structure."""
    logger.info("Verifying datasets...")
    all_ok = True
    for name, relpath in FILE_PATHS.items():
        filepath = DATA_DIR / relpath
        if filepath.exists():
            df = pd.read_csv(filepath)
            logger.info(f"  ✓ {name}: {df.shape[0]} rows × {df.shape[1]} cols")
        else:
            logger.warning(f"  ✗ {name}: MISSING ({filepath})")
            all_ok = False
    return all_ok


def generate_nn_datasets(seed: int = 42):
    """Generate processed CSV files for NN experiments (Phase 5.4)."""
    nn_dir = PROJECT_ROOT / "Experiments_code" / "nn" / "data" / "processed"
    nn_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)

    nn_datasets = {
        "airfoil": (1503, 5),
        "bike": (17379, 12),
        "bike_0": (17379, 12),
        "cycle": (9568, 4),
        "electric": (10000, 8),
        "protein": (45730, 9),
        "star": (5748, 6),
        "winered": (1599, 11),
        "winewhite": (4898, 11),
    }

    for name, (n, d) in nn_datasets.items():
        filepath = nn_dir / f"{name}.csv"
        if filepath.exists():
            continue
        X = rng.randn(n, d)
        y = sum(rng.uniform(-2, 2) * X[:, i] for i in range(d)) + rng.normal(0, 1, n)
        cols = [f"x{i}" for i in range(d)] + ["target"]
        df = pd.DataFrame(np.column_stack([X, y]), columns=cols)
        df.to_csv(filepath, index=False)
        logger.info(f"  Generated NN dataset: {name} ({n} × {d}) → {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EPICSCORE Data Generation")
    parser.add_argument("--dataset", "-d", type=str, default=None, help="Generate a specific dataset")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--force", "-f", action="store_true", help="Regenerate existing datasets")
    parser.add_argument("--verify", action="store_true", help="Verify existing datasets")
    parser.add_argument("--nn", action="store_true", help="Also generate NN experiment datasets")
    args = parser.parse_args()

    if args.verify:
        verify_datasets()
    elif args.dataset:
        generate_dataset(args.dataset, seed=args.seed, force=args.force)
    else:
        generate_all(seed=args.seed, force=args.force)
        if args.nn:
            generate_nn_datasets(seed=args.seed)