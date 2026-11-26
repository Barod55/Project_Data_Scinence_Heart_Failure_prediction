"""Shared feature schema for train.py and main.py.

The categorical codes below must stay identical to the codes used by the
HTML form (HART.html). Change them in one place only if you retrain the model
and update the form to match.
"""
import math

TARGET_COLUMN = "HeartDisease"

# Column order the model expects (same order as heart.csv, without the target).
FEATURE_COLUMNS = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]

# Alphabetical codes (what LabelEncoder produced in the old train.py).
CATEGORY_CODES = {
    "Sex": {"F": 0, "M": 1},
    "ChestPainType": {"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3},
    "RestingECG": {"LVH": 0, "Normal": 1, "ST": 2},
    "ExerciseAngina": {"N": 0, "Y": 1},
    "ST_Slope": {"Down": 0, "Flat": 1, "Up": 2},
}

# Numeric columns that are already coded 0/1 in heart.csv.
BINARY_FEATURES = ["FastingBS"]

# In heart.csv a 0 in these columns means "not measured", not a real value.
ZERO_MEANS_MISSING = ["Cholesterol", "RestingBP"]

# Values the API accepts (same limits as the min/max attributes in HART.html).
NUMERIC_LIMITS = {
    "Age": (1, 120),
    "RestingBP": (50, 250),
    "Cholesterol": (50, 700),
    "MaxHR": (40, 250),
    "Oldpeak": (-5, 10),
}


def validate_features(values):
    """Return a list of readable problems (an empty list means the input is valid)."""
    if len(values) != len(FEATURE_COLUMNS):
        return [f"expected {len(FEATURE_COLUMNS)} values, got {len(values)}"]

    problems = []
    for name, value in zip(FEATURE_COLUMNS, values):
        if not math.isfinite(value):
            problems.append(f"{name} must be a finite number")
        elif name in CATEGORY_CODES:
            allowed = sorted(CATEGORY_CODES[name].values())
            if value not in allowed:
                problems.append(f"{name} must be one of {allowed}")
        elif name in BINARY_FEATURES:
            if value not in (0, 1):
                problems.append(f"{name} must be 0 or 1")
        else:
            low, high = NUMERIC_LIMITS[name]
            if not low <= value <= high:
                problems.append(f"{name} must be between {low} and {high}")
    return problems
