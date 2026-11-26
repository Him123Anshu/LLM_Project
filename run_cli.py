"""
Command-Line Interface (CLI) Demo Runner for Loan Eligibility Agent.
Runs evaluations on sample approved and rejected applicants from LendingClub,
displaying full decisions, factor attributions, and plain-language explanations.
"""

import sys
from src.agent import LoanEligibilityAgent
from src.sample_decisions import SAMPLE_PERSONAS


def main():
    print("=" * 85)
    print("      LENDINGCLUB LOAN ELIGIBILITY AGENT - CLI DEMONSTRATION")
    print("    Dataset: LendingClub Loan Data (Kaggle) | Model: Calibrated Credit Scorecard")
    print("=" * 85)

    agent = LoanEligibilityAgent()

    for idx, (key, persona) in enumerate(SAMPLE_PERSONAS.items(), 1):
        print(f"\n[{idx}/5] CASE STUDY: {persona['name'].upper()} ({persona['description']})")
        print("-" * 85)
        print(f"  * Profile: Income=${persona['annual_inc']:,.0f} | Requested=${persona['loan_amnt']:,.0f} ({persona['term']})")
        print(f"  * Credit : FICO={int(persona['fico_score'])} | DTI={persona['dti']:.1f}% | Util={persona['revol_util']:.1f}% | Delinq(2y)={persona['delinq_2yrs']}")

        result = agent.evaluate_applicant(persona)

        status_symbol = "[APPROVED]" if result["decision"] == "APPROVED" else "[DECLINED]"
        print(f"\n  >>> DECISION: {status_symbol} {result['decision']} ({result['decision_type']})")
        print(f"  >>> METRICS : Confidence={result['approval_probability']*100:.1f}% | Grade={result['grade']} | Est. APR={result['estimated_apr']:.2f}% | Risk Score={result['risk_score']}")

        exp = result["plain_language_explanation"]
        print(f"\n  [PLAIN-LANGUAGE EXECUTIVE SUMMARY]")
        print(f"  {exp['summary']}")

        print(f"\n  [KEY FACTORS DRIVING THIS DECISION]")
        for f in exp["primary_driving_factors"]:
            print(f"    * {f['factor']} ({f['impact']}):")
            print(f"      {f['explanation']}")

        if result.get("policy_violations"):
            print(f"\n  [POLICY GATE VIOLATION]")
            for pv in result["policy_violations"]:
                print(f"    ! {pv}")

        print(f"\n  [ADVICE & REMEDIATION]")
        print(f"  {exp['next_steps_advice']}")

        if exp["counterfactual_targets"]:
            print(f"\n  [COUNTERFACTUAL WHAT-IF TARGETS]")
            for cf in exp["counterfactual_targets"]:
                print(f"    -> {cf['action']}: Currently {cf['current']} -> Target {cf['target']}")
                print(f"       Impact: {cf['impact']}")
                print(f"       Action: {cf['how_to_achieve']}")

        print("=" * 85)

    print("\nDemonstration complete. To launch the interactive web demo, run:")
    print("   streamlit run app.py\n")


if __name__ == "__main__":
    main()
