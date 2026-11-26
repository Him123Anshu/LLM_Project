"""
Explainability & Factor Attribution Engine for Loan Eligibility.
Computes exact feature attributions, adverse action reasons, and plain-language narratives.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from src.data_pipeline import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES_FRIENDLY
)
from src.model import LoanEligibilityModel


class CreditExplainer:
    """Interpretable Attribution and Plain-Language Explanation Engine."""

    def __init__(self, model: LoanEligibilityModel):
        self.model = model
        self.coef = model.model.coef_[0]
        self.intercept = model.model.intercept_[0]
        self.feature_names = model.feature_names
        self.means = model.feature_means
        self.stds = model.feature_stds

    def explain_decision(self, applicant_data: Dict[str, Any], scoring_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompose the model decision into exact feature contributions
        and generate plain-language explanations.
        """
        raw_transformed = scoring_result["raw_transformed"]
        is_eligible = scoring_result["is_eligible"]
        prob = scoring_result["approval_probability"]

        # Calculate exact log-odds contribution per transformed feature
        # Delta z_i = w_i * x_i
        # Since features were standardized: x_i = (val - mean) / std
        # A positive contribution increases approval probability; negative decreases it.
        contributions_raw = self.coef * raw_transformed

        # Aggregate contributions back to root features (e.g. combining one-hot columns)
        feature_impacts: Dict[str, float] = {}

        # 1. Numerical features
        for i, col in enumerate(NUMERICAL_FEATURES):
            feature_impacts[col] = float(contributions_raw[i])

        # 2. Categorical features (aggregate dummy variables)
        cat_start_idx = len(NUMERICAL_FEATURES)
        cat_features_encoded = self.feature_names[cat_start_idx:]

        for cat_col in CATEGORICAL_FEATURES:
            feature_impacts[cat_col] = 0.0

        for idx, enc_name in enumerate(cat_features_encoded):
            full_idx = cat_start_idx + idx
            impact = float(contributions_raw[full_idx])
            for cat_col in CATEGORICAL_FEATURES:
                if enc_name.startswith(cat_col + "_"):
                    feature_impacts[cat_col] += impact

        # Calculate total absolute impact for percentage weighting
        total_abs_impact = sum(abs(v) for v in feature_impacts.values())
        if total_abs_impact == 0:
            total_abs_impact = 1.0

        # Build structured factor list
        factors = []
        for feat, log_odds in feature_impacts.items():
            impact_pct = (abs(log_odds) / total_abs_impact) * 100.0
            direction = "positive" if log_odds > 0 else "negative"
            friendly_name = FEATURE_NAMES_FRIENDLY.get(feat, feat.replace("_", " ").title())
            applicant_val = applicant_data.get(feat, scoring_result["df_clean"][feat].values[0] if feat in scoring_result["df_clean"].columns else "N/A")

            factors.append({
                "feature": feat,
                "friendly_name": friendly_name,
                "applicant_value": applicant_val,
                "benchmark_value": self.means.get(feat, "N/A"),
                "log_odds_impact": round(log_odds, 4),
                "impact_pct": round(impact_pct, 1),
                "direction": direction,
                "plain_narrative": self._generate_feature_narrative(feat, applicant_val, log_odds, applicant_data)
            })

        # Sort factors by impact magnitude
        factors_sorted = sorted(factors, key=lambda x: abs(x["log_odds_impact"]), reverse=True)

        top_positive_factors = [f for f in factors_sorted if f["direction"] == "positive"]
        top_negative_factors = [f for f in factors_sorted if f["direction"] == "negative"]

        # Adverse Action Reasons (for denied or subprime loans, top negative drivers)
        adverse_action_reasons = [
            {
                "rank": i + 1,
                "factor": f["friendly_name"],
                "value": f["applicant_value"],
                "impact": f"-{f['impact_pct']}% penalty",
                "explanation": f["plain_narrative"]
            }
            for i, f in enumerate(top_negative_factors[:4])
        ]

        # Favorable Approval Reasons (for approved loans, top positive drivers)
        favorable_reasons = [
            {
                "rank": i + 1,
                "factor": f["friendly_name"],
                "value": f["applicant_value"],
                "impact": f"+{f['impact_pct']}% boost",
                "explanation": f["plain_narrative"]
            }
            for i, f in enumerate(top_positive_factors[:4])
        ]

        # Calculate Counterfactual suggestions for rejected applicants
        counterfactuals = self._generate_counterfactuals(applicant_data, scoring_result)

        return {
            "all_factors": factors_sorted,
            "top_positive_factors": top_positive_factors,
            "top_negative_factors": top_negative_factors,
            "adverse_action_reasons": adverse_action_reasons,
            "favorable_reasons": favorable_reasons,
            "counterfactuals": counterfactuals
        }

    def _generate_feature_narrative(self, feat: str, val: Any, log_odds: float, data: Dict[str, Any]) -> str:
        """Generate human-friendly contextual narrative for a given factor."""
        is_pos = log_odds > 0

        if feat == "fico_score":
            score = float(val) if val is not None else 690.0
            if score >= 750:
                return f"Excellent credit score of {int(score)} puts you in the top tier of borrowers, demonstrating a rock-solid history of debt management."
            elif score >= 700:
                return f"Good credit score of {int(score)} reflects consistent reliability and strong repayment behavior."
            elif score >= 640:
                return f"Fair credit score of {int(score)} meets basic thresholds but indicates moderate risk compared to prime benchmarks."
            else:
                return f"Subprime credit score of {int(score)} falls below conventional credit tiers, indicating elevated historical credit risk."

        elif feat == "dti":
            dti_val = float(val) if val is not None else 20.0
            if dti_val <= 15.0:
                return f"Very low debt-to-income ratio ({dti_val:.1f}%) means your existing debt payments take up very little of your income, leaving high capacity for repayment."
            elif dti_val <= 28.0:
                return f"Healthy debt-to-income ratio ({dti_val:.1f}%) is well within prudent lending guidelines (below the 36% ceiling)."
            elif dti_val <= 38.0:
                return f"Elevated debt-to-income ratio ({dti_val:.1f}%) indicates that existing obligations consume a substantial share of your monthly cash flow."
            else:
                return f"Critical debt-to-income ratio ({dti_val:.1f}%) exceeds prudent underwriting guidelines, indicating severe debt burden."

        elif feat == "annual_inc":
            inc = float(val) if val is not None else 60000.0
            if is_pos:
                return f"Robust annual income of ${inc:,.0f} provides strong debt service coverage and cash-flow cushion."
            else:
                return f"Annual income of ${inc:,.0f} is tight relative to the requested loan size and living expenses."

        elif feat == "loan_amnt" or feat == "loan_to_income":
            loan = float(data.get("loan_amnt", 15000))
            inc = max(float(data.get("annual_inc", 60000)), 1.0)
            ratio = (loan / inc) * 100
            if is_pos:
                return f"Requested loan amount (${loan:,.0f}) is modest ({ratio:.1f}% of income), keeping monthly installments very manageable."
            else:
                return f"Requested loan of ${loan:,.0f} represents {ratio:.1f}% of your annual income, creating a heavy repayment commitment."

        elif feat == "revol_util":
            util = float(val) if val is not None else 50.0
            if util <= 25.0:
                return f"Low credit card utilization of {util:.1f}% shows you maintain significant unused credit cushion."
            elif util <= 50.0:
                return f"Moderate credit card utilization of {util:.1f}% is acceptable, though lowering it below 30% would optimize your profile."
            else:
                return f"High revolving line utilization ({util:.1f}%) indicates over-reliance on revolving credit lines."

        elif feat == "delinq_2yrs":
            count = int(val) if val is not None else 0
            if count == 0:
                return "Clean payment history with 0 late payments (30+ days overdue) over the last 2 years."
            else:
                return f"{count} recent delinquency incident(s) in the past 24 months flagged as high repayment risk."

        elif feat == "inq_last_6mths":
            inq = int(val) if val is not None else 0
            if inq == 0:
                return "No recent hard credit inquiries in the past 6 months, indicating stable credit seeking."
            elif inq <= 2:
                return f"{inq} recent credit inquiry/inquiries, which is within standard credit-seeking behavior."
            else:
                return f"{inq} hard inquiries in the past 6 months signals frequent recent credit applications."

        elif feat == "pub_rec_bankruptcies":
            bk = int(val) if val is not None else 0
            if bk == 0:
                return "Clean public legal records with zero bankruptcies on file."
            else:
                return f"{bk} public record bankruptcy on file, which severely restricts credit eligibility under lending policy."

        elif feat == "emp_length_years":
            yrs = float(val) if val is not None else 3.0
            if yrs >= 5:
                return f"Strong employment tenure of {yrs:.0f}+ years indicates career stability and reliable ongoing income."
            else:
                return f"Employment tenure of {yrs:.0f} year(s) represents an emerging employment history."

        elif feat == "home_ownership":
            home = str(val).upper()
            if home == "MORTGAGE":
                return "Homeowner with established mortgage demonstrates residential stability and asset backing."
            elif home == "OWN":
                return "Outright home ownership provides excellent asset stability with no monthly rent/mortgage obligation."
            else:
                return "Renting status provides flexibility but offers no real estate equity backing."

        elif feat == "purpose":
            purp = str(val).replace("_", " ").title()
            if "Debt Consolidation" in purp or "Credit Card" in purp:
                return f"Loan purpose ({purp}) aims to restructure higher-interest revolving debt into a structured fixed installment."
            else:
                return f"Loan purpose: {purp}."

        return f"{feat.replace('_', ' ').title()} value of {val} {'positively contributed' if is_pos else 'negatively impacted'} the decision."

    def _generate_counterfactuals(self, applicant_data: Dict[str, Any], scoring_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate actionable 'what-if' counterfactual adjustments to flip a decision or improve grade."""
        is_eligible = scoring_result["is_eligible"]
        prob = scoring_result["approval_probability"]
        actions = []

        fico = float(applicant_data.get("fico_score", 680))
        dti = float(applicant_data.get("dti", 25))
        loan = float(applicant_data.get("loan_amnt", 15000))
        inc = float(applicant_data.get("annual_inc", 60000))
        util = float(applicant_data.get("revol_util", 55))

        if not is_eligible:
            # Action 1: FICO improvement target
            target_fico = min(850, int(fico + max(35, (0.60 - prob) * 150)))
            actions.append({
                "action": "Improve Credit Score (FICO)",
                "current": f"{int(fico)}",
                "target": f"{target_fico}+",
                "impact": "Boosts approval probability past the 55% eligibility threshold.",
                "how_to_achieve": "Ensure 100% on-time payments for 6 consecutive months and dispute any inaccurate bureau marks."
            })

            # Action 2: DTI Reduction target
            if dti > 25.0:
                target_dti = max(18.0, round(dti * 0.70, 1))
                monthly_debt_reduction = round((dti - target_dti) / 100.0 * (inc / 12.0))
                actions.append({
                    "action": "Lower Debt-to-Income (DTI)",
                    "current": f"{dti:.1f}%",
                    "target": f"<= {target_dti:.1f}%",
                    "impact": "Decreases monthly cash outflow strain.",
                    "how_to_achieve": f"Pay down approximately ${monthly_debt_reduction:,.0f}/month in existing credit cards or auto loans."
                })

            # Action 3: Credit Card Utilization
            if util > 35.0:
                actions.append({
                    "action": "Pay Down Revolving Balances",
                    "current": f"{util:.1f}% utilization",
                    "target": "<= 28.0% utilization",
                    "impact": "Quickest way to gain 20-40 FICO points within 30-60 days.",
                    "how_to_achieve": "Pay down credit card balances before statement closing dates."
                })

            # Action 4: Counter-offer on Loan Amount
            if loan > 10000 and (loan / inc) > 0.25:
                affordable_loan = round(inc * 0.18, -2)
                actions.append({
                    "action": "Request a Smaller Loan Amount (Counter-Offer)",
                    "current": f"${loan:,.0f}",
                    "target": f"${affordable_loan:,.0f}",
                    "impact": "Immediately reduces the installment-to-income ratio to an approvable level.",
                    "how_to_achieve": "Re-submit application for the adjusted principal amount."
                })
        else:
            # For approved applicants: how to qualify for Grade A or lower APR
            if scoring_result["grade"] != "A":
                actions.append({
                    "action": "Upgrade to Grade A / Lowest APR Tier",
                    "current": f"Grade {scoring_result['grade']} ({scoring_result['estimated_apr']}%)",
                    "target": "Grade A (Under 8.5% APR)",
                    "impact": "Save thousands of dollars in interest over the life of the loan.",
                    "how_to_achieve": "Maintain revolving utilization below 15% and avoid new credit inquiries for 12 months."
                })

        return actions
