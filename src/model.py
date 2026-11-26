"""
Machine Learning Credit Model for LendingClub Loan Eligibility.
Provides training, scoring, risk grade mapping, and evaluation metrics.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from src.data_pipeline import (
    build_preprocessor,
    get_train_test_data,
    preprocess_dataframe,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "loan_model.joblib")


class LoanEligibilityModel:
    """Production credit risk scoring model for LendingClub loan underwriting."""

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold
        self.preprocessor = None
        self.model = None
        self.feature_names = []
        self.feature_means = {}
        self.feature_stds = {}
        self.metrics = {}
        self.is_trained = False

    def train(self, data_path: str = "data/lending_club_sample.csv") -> Dict[str, Any]:
        """Train the credit underwriting model and compute benchmark statistics."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        X_train, X_test, y_train, y_test = get_train_test_data(data_path)

        # Store baseline population statistics for explanation reference
        for col in NUMERICAL_FEATURES:
            self.feature_means[col] = float(X_train[col].mean())
            self.feature_stds[col] = float(X_train[col].std())

        # Fit preprocessor
        self.preprocessor = build_preprocessor()
        X_train_trans = self.preprocessor.fit_transform(X_train)
        X_test_trans = self.preprocessor.transform(X_test)

        # Get feature names after one-hot encoding
        cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_features_encoded = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
        self.feature_names = NUMERICAL_FEATURES + list(cat_features_encoded)

        # Train calibrated Logistic Regression credit scorecard
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs"
        )
        self.model.fit(X_train_trans, y_train)

        # Evaluate performance on held-out test split
        y_prob = self.model.predict_proba(X_test_trans)[:, 1]
        y_pred = (y_prob >= self.threshold).astype(int)

        self.metrics = {
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "n_train": len(X_train),
            "n_test": len(X_test)
        }

        self.is_trained = True
        self.save()
        return self.metrics

    def predict_score(self, applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate an applicant profile.
        Returns:
            - is_eligible (bool)
            - approval_probability (float)
            - risk_score (int: 300 to 850 scale)
            - assigned_grade (A - G)
            - estimated_apr (float)
            - raw_transformed_features
        """
        if not self.is_trained:
            self.load()

        df_single = pd.DataFrame([applicant_data])
        df_clean = preprocess_dataframe(df_single)
        X_trans = self.preprocessor.transform(df_clean[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])

        prob_approval = float(self.model.predict_proba(X_trans)[0, 1])
        is_eligible = bool(prob_approval >= self.threshold)

        # Map approval probability to a calibrated Credit Bureau style score (300 - 850)
        credit_score = int(np.clip(300 + prob_approval * 550, 300, 850))

        # Assign LendingClub risk grade & estimated APR
        if prob_approval >= 0.88:
            grade = "A"
            apr = 6.99 + (0.95 - prob_approval) * 20.0
        elif prob_approval >= 0.76:
            grade = "B"
            apr = 10.49 + (0.88 - prob_approval) * 25.0
        elif prob_approval >= 0.64:
            grade = "C"
            apr = 14.25 + (0.76 - prob_approval) * 30.0
        elif prob_approval >= 0.50:
            grade = "D"
            apr = 18.50 + (0.64 - prob_approval) * 35.0
        elif prob_approval >= 0.38:
            grade = "E"
            apr = 23.00 + (0.50 - prob_approval) * 40.0
        elif prob_approval >= 0.25:
            grade = "F"
            apr = 27.50 + (0.38 - prob_approval) * 45.0
        else:
            grade = "G"
            apr = 31.99

        apr = round(float(np.clip(apr, 5.49, 34.99)), 2)

        return {
            "is_eligible": is_eligible,
            "approval_probability": round(prob_approval, 4),
            "credit_score": credit_score,
            "grade": grade,
            "estimated_apr": apr,
            "raw_transformed": X_trans[0],
            "df_clean": df_clean
        }

    def save(self, filepath: str = MODEL_PATH):
        """Serialize model bundle."""
        state = {
            "threshold": self.threshold,
            "preprocessor": self.preprocessor,
            "model": self.model,
            "feature_names": self.feature_names,
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
            "metrics": self.metrics,
            "is_trained": self.is_trained
        }
        joblib.dump(state, filepath)

    @classmethod
    def load(cls, filepath: str = MODEL_PATH) -> "LoanEligibilityModel":
        """Deserialize model bundle or train if missing."""
        if os.path.exists(filepath):
            state = joblib.load(filepath)
            # If pickled as instance previously, or dict
            if isinstance(state, dict):
                model_inst = cls(threshold=state.get("threshold", 0.55))
                model_inst.preprocessor = state.get("preprocessor")
                model_inst.model = state.get("model")
                model_inst.feature_names = state.get("feature_names", [])
                model_inst.feature_means = state.get("feature_means", {})
                model_inst.feature_stds = state.get("feature_stds", {})
                model_inst.metrics = state.get("metrics", {})
                model_inst.is_trained = state.get("is_trained", True)
                return model_inst
            else:
                return state
        model = cls()
        model.train()
        return model


def get_trained_model() -> LoanEligibilityModel:
    """Singleton getter for trained model."""
    if os.path.exists(MODEL_PATH):
        return LoanEligibilityModel.load(MODEL_PATH)
    model = LoanEligibilityModel()
    model.train()
    return model


if __name__ == "__main__":
    model = LoanEligibilityModel()
    metrics = model.train()
    print("Model Training Complete!")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix: {metrics['confusion_matrix']}")
