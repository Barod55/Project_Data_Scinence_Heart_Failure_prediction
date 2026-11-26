"""FastAPI backend for the heart-failure risk form.

Run:   fastapi dev app.py           (or: uvicorn app:app --reload)
Open:  http://127.0.0.1:8000/       (the form is served from the same address)

Train the model first with:  python train.py
"""
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from features import FEATURE_COLUMNS, validate_features

logger = logging.getLogger("heart-api")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
UI_PATH = BASE_DIR / "HART.html"

RESULT_LABELS = {1: "Higher risk predicted", 0: "Lower risk predicted"}


def load_model(path: Path):
    """Load models/model.joblib and check that it matches the feature schema."""
    if not path.exists():
        raise RuntimeError(f"{path} not found. Run `python train.py` first.")

    artifact = joblib.load(path)
    if isinstance(artifact, dict):
        model = artifact["model"]
        columns = list(artifact.get("feature_columns", FEATURE_COLUMNS))
        trained_with = artifact.get("sklearn_version")
    else:
        model = artifact
        columns = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
        trained_with = None

    if trained_with and trained_with != sklearn.__version__:
        logger.warning(
            "Model was trained with scikit-learn %s but %s is installed; "
            "retrain or install the matching version.",
            trained_with,
            sklearn.__version__,
        )
    if columns != FEATURE_COLUMNS:
        raise RuntimeError(
            "Model feature columns do not match features.FEATURE_COLUMNS: "
            f"{columns} vs {FEATURE_COLUMNS}"
        )
    return model


model = load_model(MODEL_PATH)
POSITIVE_INDEX = list(model.classes_).index(1)

app = FastAPI(title="Heart Failure Prediction API")

# "*" is fine for local use. When deploying, list the real origins, e.g.
#   ALLOWED_ORIGINS=https://example.org,https://app.example.org
allowed_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientInput(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=len(FEATURE_COLUMNS),
        max_length=len(FEATURE_COLUMNS),
        description=f"Values in this order: {', '.join(FEATURE_COLUMNS)}",
    )

    @field_validator("features")
    @classmethod
    def check_features(cls, values):
        problems = validate_features(values)
        if problems:
            raise ValueError("; ".join(problems))
        return values


class PredictionOut(BaseModel):
    prediction: int
    result: str
    risk_probability: float  # 0.0 - 1.0, predicted probability of heart disease


@app.get("/")
def serve_ui():
    if not UI_PATH.exists():
        raise HTTPException(status_code=404, detail=f"{UI_PATH.name} not found next to app.py")
    return FileResponse(UI_PATH)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Heart Failure Prediction API is running",
        "model_loaded": True,
        "sklearn_version": sklearn.__version__,
    }


@app.post("/predict", response_model=PredictionOut)
def predict(data: PatientInput):
    df_input = pd.DataFrame([data.features], columns=FEATURE_COLUMNS)

    pred = int(model.predict(df_input)[0])
    prob = float(model.predict_proba(df_input)[0][POSITIVE_INDEX])

    return PredictionOut(
        prediction=pred,
        result=RESULT_LABELS[pred],
        risk_probability=round(prob, 4),
    )