"""
Batch Underwriting Pipeline.
Processes an input CSV of loan applicants, runs the autonomous Underwriting Agent,
and outputs a comprehensive decision and explanation audit file.
"""

import sys
import os
import argparse
import pandas as pd
from typing import Optional
from src.agent import LoanEligibilityAgent
from src.data_pipeline import preprocess_dataframe


def run_batch_underwriting(input_csv: str, output_csv: Optional[str] = None) -> pd.DataFrame:
    """Evaluate an entire batch of loan applicants and export structured explanations."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df_in = pd.read_csv(input_csv)
    print(f"Loaded {len(df_in)} loan applicants from '{input_csv}'.")

    agent = LoanEligibilityAgent()
    records = df_in.to_dict(orient="records")
    results = []

    print("Running underwriting agent and factor attribution engine...")
    for idx, row in enumerate(records, 1):
        eval_res = agent.evaluate_applicant(row)
        exp = eval_res["plain_language_explanation"]
        drv_factors = exp["primary_driving_factors"]

        f1 = drv_factors[0] if len(drv_factors) > 0 else {"factor": "N/A", "impact": "0%", "explanation": "N/A"}
        f2 = drv_factors[1] if len(drv_factors) > 1 else {"factor": "N/A", "impact": "0%", "explanation": "N/A"}
        f3 = drv_factors[2] if len(drv_factors) > 2 else {"factor": "N/A", "impact": "0%", "explanation": "N/A"}

        cf_list = exp.get("counterfactual_targets", [])
        cf_summary = f"{cf_list[0]['action']}: {cf_list[0]['current']} -> {cf_list[0]['target']}" if cf_list else "None (Prime Grade)"

        res_row = dict(row)
        res_row.update({
            "decision": eval_res["decision"],
            "decision_type": eval_res["decision_type"],
            "approval_probability": round(eval_res["approval_probability"], 4),
            "lending_club_grade": eval_res["grade"],
            "estimated_apr": eval_res["estimated_apr"],
            "risk_score": eval_res["risk_score"],
            "policy_violations": "; ".join(eval_res.get("policy_violations", [])) if eval_res.get("policy_violations") else "None",
            "executive_summary": exp["summary"],
            "top_factor_1_name": f1["factor"],
            "top_factor_1_impact": f1["impact"],
            "top_factor_1_explanation": f1["explanation"],
            "top_factor_2_name": f2["factor"],
            "top_factor_2_impact": f2["impact"],
            "top_factor_2_explanation": f2["explanation"],
            "top_factor_3_name": f3["factor"],
            "top_factor_3_impact": f3["impact"],
            "top_factor_3_explanation": f3["explanation"],
            "counterfactual_target": cf_summary
        })
        results.append(res_row)

    df_out = pd.DataFrame(results)

    if output_csv is None:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_underwritten{ext}"

    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Batch underwriting complete! Enriched audit file saved to: '{output_csv}'")

    # Print summary statistics
    n_app = (df_out["decision"] == "APPROVED").sum()
    n_rej = (df_out["decision"] == "REJECTED").sum()
    print("\n" + "=" * 60)
    print(f"BATCH SUMMARY: Total={len(df_out)} | Approved={n_app} ({n_app/len(df_out):.1%}) | Declined={n_rej} ({n_rej/len(df_out):.1%})")
    print("Grade Breakdown:")
    print(df_out["lending_club_grade"].value_counts().sort_index().to_string())
    print("=" * 60 + "\n")

    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Underwriting CLI for LendingClub Loan Eligibility Agent")
    parser.add_argument("--input", default="data/sample_batch_applicants.csv", help="Input CSV path with applicant profiles")
    parser.add_argument("--output", default="data/sample_batch_underwritten.csv", help="Output CSV path for underwritten results")
    args = parser.parse_args()

    # If sample batch file doesn't exist, create one from sample personas + random records
    if not os.path.exists(args.input):
        print(f"Generating sample input file at '{args.input}' for demonstration...")
        from src.sample_decisions import SAMPLE_PERSONAS
        sample_rows = list(SAMPLE_PERSONAS.values())
        pd.DataFrame(sample_rows).to_csv(args.input, index=False)

    run_batch_underwriting(args.input, args.output)
