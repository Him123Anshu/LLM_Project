"""
Data Preprocessing and Pipeline for LendingClub Loan Eligibility.
Handles feature extraction, scaling, encoding, and dataset preparation.
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


NUMERICAL_FEATURES = [
    "annual_inc",
    "loan_amnt",
    "dti",
    "fico_score",
    "emp_length_years",
    "revol_util",
    "revol_bal",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "total_acc",
    "pub_rec_bankruptcies",
    "loan_to_income"
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "purpose",
    "term"
]

FEATURE_NAMES_FRIENDLY = {
    "annual_inc": "Annual Income ($)",
    "loan_amnt": "Loan Amount Requested ($)",
    "dti": "Debt-to-Income Ratio (%)",
    "fico_score": "Credit Score (FICO)",
    "emp_length_years": "Employment Length (Years)",
    "revol_util": "Revolving Credit Utilization (%)",
    "revol_bal": "Revolving Credit Balance ($)",
    "delinq_2yrs": "Past 2-Year Delinquencies",
    "inq_last_6mths": "Recent Inquiries (6 Mos)",
    "open_acc": "Open Credit Lines",
    "total_acc": "Total Credit Accounts",
    "pub_rec_bankruptcies": "Public Record Bankruptcies",
    "loan_to_income": "Loan-to-Income Ratio",
    "home_ownership": "Home Ownership",
    "purpose": "Loan Purpose",
    "term": "Loan Term"
}


def parse_emp_length(val: Any) -> float:
    """Parse emp_length string like '10+ years' or '< 1 year' to numerical float."""
    if pd.isna(val):
        return 0.0
    val_str = str(val).strip()
    if "<" in val_str:
        return 0.5
    match = re.search(r"\d+", val_str)
    if match:
        return float(match.group(0))
    return 0.0


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize raw dataframe and engineer required features."""
    df_clean = df.copy()

    # Calculate emp_length_years
    if "emp_length_years" not in df_clean.columns and "emp_length" in df_clean.columns:
        df_clean["emp_length_years"] = df_clean["emp_length"].apply(parse_emp_length)
    elif "emp_length_years" not in df_clean.columns:
        df_clean["emp_length_years"] = 3.0

    # Ensure annual_inc is positive
    df_clean["annual_inc"] = np.maximum(df_clean["annual_inc"], 1000.0)

    # Calculate loan_to_income
    df_clean["loan_to_income"] = (df_clean["loan_amnt"] / df_clean["annual_inc"]).round(4)

    # Fill missing values if any
    defaults = {
        "annual_inc": 65000.0,
        "loan_amnt": 15000.0,
        "dti": 18.0,
        "fico_score": 690.0,
        "emp_length_years": 4.0,
        "revol_util": 50.0,
        "revol_bal": 12000.0,
        "delinq_2yrs": 0,
        "inq_last_6mths": 0,
        "open_acc": 10,
        "total_acc": 22,
        "pub_rec_bankruptcies": 0,
        "loan_to_income": 0.23,
        "home_ownership": "RENT",
        "purpose": "debt_consolidation",
        "term": "36 months"
    }

    for col, val in defaults.items():
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(val)

    # Make sure term is clean string
    if "term" in df_clean.columns:
        df_clean["term"] = df_clean["term"].astype(str).str.strip()

    return df_clean


def build_preprocessor() -> ColumnTransformer:
    """Build scikit-learn ColumnTransformer for numerical and categorical features."""
    num_pipe = Pipeline([
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, NUMERICAL_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )
    return preprocessor


def get_train_test_data(data_path: str = "data/lending_club_sample.csv") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, preprocess, and split LendingClub dataset."""
    from data.generate_data import ensure_lending_club_data

    df = ensure_lending_club_data(data_path)
    df = preprocess_dataframe(df)

    # Target: is_eligible (1 = Approved / Fully Paid, 0 = High Risk / Default)
    if "is_eligible" in df.columns:
        y = df["is_eligible"]
    elif "loan_status" in df.columns:
        y = (df["loan_status"] == "Fully Paid").astype(int)
    else:
        raise ValueError("Target column is_eligible or loan_status not found in dataset.")

    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
