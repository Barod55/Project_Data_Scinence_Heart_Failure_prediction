from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart.csv"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

FEATURE_COLUMNS = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # نفس ترميز الـ Notebook (مطابق لـ HART.html)
    cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # تحويل الأصفار غير المنطقية في الضغط والكوليسترول إلى NaN لمعالجتها بالوسيط داخل الـ Pipeline
    for col in ["RestingBP", "Cholesterol"]:
        df[col] = df[col].replace(0, np.nan)

    X = df[FEATURE_COLUMNS]
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    stacking_model = StackingClassifier(
        estimators=[
            (
                "cat",
                CatBoostClassifier(
                    verbose=0,
                    random_state=42,
                    allow_writing_files=False,
                ),
            ),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            ("svm", SVC(probability=True, random_state=42)),
        ],
        final_estimator=LogisticRegression(),
        cv=5,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", stacking_model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    print(f"Stacking Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(f"Stacking ROC-AUC:  {roc_auc_score(y_test, y_prob) * 100:.2f}%")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()