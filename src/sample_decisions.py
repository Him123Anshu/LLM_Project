"""
Sample Decision Scenarios and Persona Profiles.
Provides realistic Approved, Rejected, and Borderline case studies with full agent explanations.
"""

from typing import Dict, Any, List
from src.agent import LoanEligibilityAgent


SAMPLE_PERSONAS: Dict[str, Dict[str, Any]] = {
    "approved_prime": {
        "applicant_id": "APP-98214 (Sarah Jenkins)",
        "name": "Sarah Jenkins",
        "description": "Prime Borrower - High Income, Low DTI, Excellent FICO",
        "annual_inc": 115000.0,
        "loan_amnt": 20000.0,
        "term": "36 months",
        "fico_score": 775.0,
        "dti": 11.2,
        "emp_length_years": 8.0,
        "home_ownership": "MORTGAGE",
        "purpose": "home_improvement",
        "delinq_2yrs": 0,
        "inq_last_6mths": 0,
        "open_acc": 14,
        "total_acc": 28,
        "revol_bal": 6200.0,
        "revol_util": 14.5,
        "pub_rec_bankruptcies": 0
    },
    "approved_near_prime": {
        "applicant_id": "APP-84721 (Marcus Vance)",
        "name": "Marcus Vance",
        "description": "Near-Prime Borrower - Solid Income, Moderate DTI & Utilization",
        "annual_inc": 64000.0,
        "loan_amnt": 12000.0,
        "term": "36 months",
        "fico_score": 685.0,
        "dti": 21.4,
        "emp_length_years": 4.0,
        "home_ownership": "RENT",
        "purpose": "debt_consolidation",
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
        "open_acc": 9,
        "total_acc": 18,
        "revol_bal": 11400.0,
        "revol_util": 41.0,
        "pub_rec_bankruptcies": 0
    },
    "rejected_high_debt": {
        "applicant_id": "APP-51902 (David Miller)",
        "name": "David Miller",
        "description": "Overextended Borrower - Excessive Loan-to-Income, High DTI & Utilization",
        "annual_inc": 42000.0,
        "loan_amnt": 28000.0,  # 66% of annual income!
        "term": "60 months",
        "fico_score": 622.0,
        "dti": 37.8,
        "emp_length_years": 2.0,
        "home_ownership": "RENT",
        "purpose": "debt_consolidation",
        "delinq_2yrs": 1,
        "inq_last_6mths": 3,
        "open_acc": 12,
        "total_acc": 19,
        "revol_bal": 21500.0,
        "revol_util": 84.5,
        "pub_rec_bankruptcies": 0
    },
    "rejected_adverse_history": {
        "applicant_id": "APP-32049 (Elena Rostova)",
        "name": "Elena Rostova",
        "description": "Credit Policy Decline - Low FICO Floor, Past Delinquencies & Bankruptcy",
        "annual_inc": 36000.0,
        "loan_amnt": 15000.0,
        "term": "36 months",
        "fico_score": 570.0,  # Below 580 floor
        "dti": 34.0,
        "emp_length_years": 1.0,
        "home_ownership": "RENT",
        "purpose": "credit_card",
        "delinq_2yrs": 2,
        "inq_last_6mths": 2,
        "open_acc": 7,
        "total_acc": 14,
        "revol_bal": 14200.0,
        "revol_util": 89.0,
        "pub_rec_bankruptcies": 1
    },
    "borderline_counterfactual": {
        "applicant_id": "APP-67104 (Carlos Gomez)",
        "name": "Carlos Gomez",
        "description": "Borderline Applicant - Solvable Debt Load with Counterfactual Adjustment",
        "annual_inc": 54000.0,
        "loan_amnt": 22000.0,
        "term": "36 months",
        "fico_score": 648.0,
        "dti": 29.8,
        "emp_length_years": 5.0,
        "home_ownership": "RENT",
        "purpose": "debt_consolidation",
        "delinq_2yrs": 0,
        "inq_last_6mths": 2,
        "open_acc": 11,
        "total_acc": 21,
        "revol_bal": 16800.0,
        "revol_util": 68.0,
        "pub_rec_bankruptcies": 0
    }
}


def run_sample_evaluations() -> List[Dict[str, Any]]:
    """Execute the Underwriting Agent across all sample personas."""
    agent = LoanEligibilityAgent()
    results = []

    for key, persona in SAMPLE_PERSONAS.items():
        res = agent.evaluate_applicant(persona)
        res["persona_key"] = key
        res["persona_name"] = persona["name"]
        res["persona_description"] = persona["description"]
        results.append(res)

    return results


if __name__ == "__main__":
    evaluations = run_sample_evaluations()
    for ev in evaluations:
        print("=" * 80)
        print(f"Applicant: {ev['persona_name']} | Decision: {ev['decision']} ({ev['decision_type']})")
        print(f"Risk Score: {ev['risk_score']} | Grade: {ev['grade']} | Estimated APR: {ev['estimated_apr']}%")
        print(f"Approval Probability: {ev['approval_probability'] * 100:.1f}%")
        print(f"\nHeadline: {ev['plain_language_explanation']['headline']}")
        print(f"Summary: {ev['plain_language_explanation']['summary']}")
        print("\nPrimary Driving Factors:")
        for factor in ev['plain_language_explanation']['primary_driving_factors']:
            print(f"  * {factor['factor']} ({factor['impact']}): {factor['explanation']}")
        print(f"\nAdvice / Remediation: {ev['plain_language_explanation']['next_steps_advice']}")
        if ev['plain_language_explanation']['counterfactual_targets']:
            print("Counterfactual Targets:")
            for cf in ev['plain_language_explanation']['counterfactual_targets']:
                print(f"  -> {cf['action']}: Current={cf['current']}, Target={cf['target']} ({cf['impact']})")
        print("=" * 80 + "\n")
