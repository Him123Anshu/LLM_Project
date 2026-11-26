"""
Loan Eligibility Agent Module.
Autonomous underwriting agent that evaluates loan applicants, executes credit policies,
calls the ML scoring engine, performs feature attribution, and generates plain-language explanations.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from src.model import LoanEligibilityModel, get_trained_model
from src.explainer import CreditExplainer


class UnderwritingPolicy:
    """Standard Underwriting Gate Policies matching institutional credit guidelines."""

    MIN_FICO_HARD_FLOOR = 580
    MAX_DTI_HARD_CEILING = 45.0
    MAX_BANKRUPTCIES = 1
    MIN_ANNUAL_INCOME = 15000.0

    @classmethod
    def evaluate_gates(cls, data: Dict[str, Any]) -> List[str]:
        """Check for disqualifying hard policy breaches."""
        violations = []
        fico = float(data.get("fico_score", 690))
        dti = float(data.get("dti", 20.0))
        bk = int(data.get("pub_rec_bankruptcies", 0))
        inc = float(data.get("annual_inc", 60000.0))

        if fico < cls.MIN_FICO_HARD_FLOOR:
            violations.append(
                f"Credit score ({int(fico)}) falls below the minimum institutional threshold of {cls.MIN_FICO_HARD_FLOOR}."
            )
        if dti > cls.MAX_DTI_HARD_CEILING:
            violations.append(
                f"Debt-to-Income ratio ({dti:.1f}%) exceeds the maximum allowable threshold of {cls.MAX_DTI_HARD_CEILING}%."
            )
        if bk > cls.MAX_BANKRUPTCIES:
            violations.append(
                f"Multiple public record bankruptcies ({bk}) exceed the policy limit of {cls.MAX_BANKRUPTCIES}."
            )
        if inc < cls.MIN_ANNUAL_INCOME:
            violations.append(
                f"Annual income (${inc:,.0f}) is below the minimum required income threshold of ${cls.MIN_ANNUAL_INCOME:,.0f}."
            )

        return violations


class LoanEligibilityAgent:
    """
    Intelligent Credit Underwriting Agent.
    Coordinates policy checks, machine learning scoring, factor attribution,
    and plain-language natural language generation (powered by Google Gemini LLM
    with deterministic fallback).
    """

    def __init__(self, model: Optional[LoanEligibilityModel] = None, llm_engine: Optional[Any] = None):
        self.model = model or get_trained_model()
        self.explainer = CreditExplainer(self.model)
        if llm_engine is not None:
            self.llm_engine = llm_engine
        else:
            try:
                from src.llm_engine import GeminiUnderwritingEngine
                self.llm_engine = GeminiUnderwritingEngine()
            except Exception:
                self.llm_engine = None

    def evaluate_applicant(self, applicant_data: Dict[str, Any], use_llm: bool = True) -> Dict[str, Any]:
        """
        Complete agentic underwriting workflow:
        1. Sanitize & ingest profile
        2. Evaluate hard underwriting policy gates
        3. Execute ML model risk scoring
        4. Factor attribution decomposition
        5. Generate plain-language consumer explanation & underwriter dossier
           (synthesized via Gemini LLM when active, or robust heuristic fallback)
        """
        # Step 1: Ingest & format
        profile = self._sanitize_input(applicant_data)

        # Step 2: Policy Gates
        gate_violations = UnderwritingPolicy.evaluate_gates(profile)
        policy_declined = len(gate_violations) > 0

        # Step 3: ML Model Prediction
        scoring_res = self.model.predict_score(profile)

        # Final decision combines policy check with ML threshold
        if policy_declined:
            decision = "REJECTED"
            is_eligible = False
            decision_type = "Policy & Statistical Decline"
        elif scoring_res["is_eligible"]:
            decision = "APPROVED"
            is_eligible = True
            decision_type = "Credit Model Approval"
        else:
            decision = "REJECTED"
            is_eligible = False
            decision_type = "Credit Model Risk Decline"

        # Update scoring result eligibility flag with final decision
        scoring_res["is_eligible"] = is_eligible

        # Step 4: Explainability & Factor Attribution
        explanation_res = self.explainer.explain_decision(profile, scoring_res)

        # Step 5: Plain-Language Agent Synthesis (LLM vs Deterministic Fallback)
        if use_llm and self.llm_engine and getattr(self.llm_engine, "is_active", False):
            plain_language_narrative = self.llm_engine.generate_underwriting_dossier(
                profile=profile,
                decision=decision,
                decision_type=decision_type,
                gate_violations=gate_violations,
                scoring=scoring_res,
                explanation=explanation_res,
                fallback_func=self._generate_plain_language_explanation
            )
        else:
            plain_language_narrative = self._generate_plain_language_explanation(
                profile=profile,
                decision=decision,
                decision_type=decision_type,
                gate_violations=gate_violations,
                scoring=scoring_res,
                explanation=explanation_res
            )
            plain_language_narrative["llm_enhanced"] = False
            plain_language_narrative["llm_model"] = "none (heuristic fallback)"
            plain_language_narrative["llm_narrative"] = plain_language_narrative.get("summary", "")

        return {
            "applicant_id": profile.get("applicant_id", "APP-UNKNOWN"),
            "decision": decision,
            "decision_type": decision_type,
            "is_eligible": is_eligible,
            "approval_probability": scoring_res["approval_probability"],
            "risk_score": scoring_res["credit_score"],
            "grade": scoring_res["grade"],
            "estimated_apr": scoring_res["estimated_apr"],
            "policy_violations": gate_violations,
            "plain_language_explanation": plain_language_narrative,
            "factor_attribution": explanation_res,
            "llm_enhanced": plain_language_narrative.get("llm_enhanced", False),
            "llm_model": plain_language_narrative.get("llm_model", "none"),
            "raw_profile": profile
        }

    def _sanitize_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate applicant profile fields."""
        p = dict(data)
        p["annual_inc"] = float(p.get("annual_inc", 60000.0))
        p["loan_amnt"] = float(p.get("loan_amnt", 15000.0))
        p["fico_score"] = float(p.get("fico_score", 690.0))
        p["dti"] = float(p.get("dti", 18.0))
        p["emp_length_years"] = float(p.get("emp_length_years", 3.0))
        p["revol_util"] = float(p.get("revol_util", 45.0))
        p["revol_bal"] = float(p.get("revol_bal", 10000.0))
        p["delinq_2yrs"] = int(p.get("delinq_2yrs", 0))
        p["inq_last_6mths"] = int(p.get("inq_last_6mths", 0))
        p["open_acc"] = int(p.get("open_acc", 10))
        p["total_acc"] = int(p.get("total_acc", 20))
        p["pub_rec_bankruptcies"] = int(p.get("pub_rec_bankruptcies", 0))
        p["home_ownership"] = str(p.get("home_ownership", "RENT")).upper()
        p["purpose"] = str(p.get("purpose", "debt_consolidation"))
        p["term"] = str(p.get("term", "36 months"))
        return p

    def _generate_plain_language_explanation(
        self,
        profile: Dict[str, Any],
        decision: str,
        decision_type: str,
        gate_violations: List[str],
        scoring: Dict[str, Any],
        explanation: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize quantitative factor attributions into clear, empathetic, plain-language text.
        """
        loan_amnt = profile["loan_amnt"]
        prob = scoring["approval_probability"]
        grade = scoring["grade"]
        apr = scoring["estimated_apr"]
        top_pos = explanation["top_positive_factors"]
        top_neg = explanation["top_negative_factors"]

        if decision == "APPROVED":
            headline = f"Congratulations! Your application for a ${loan_amnt:,.0f} loan has been APPROVED."
            summary = (
                f"Based on our comprehensive underwriting analysis, your credit profile demonstrates strong repayment "
                f"capacity with an approval confidence score of {prob*100:.1f}%. You have been assigned LendingClub "
                f"Grade {grade} with an estimated interest rate of {apr:.2f}% APR."
            )

            # Key driving factors for approval
            factor_highlights = []
            for item in top_pos[:3]:
                factor_highlights.append({
                    "factor": item["friendly_name"],
                    "impact": f"+{item['impact_pct']}% favorable influence",
                    "explanation": item["plain_narrative"]
                })

            recommendation = (
                f"To keep this favorable rate and maintain a prime credit tier, we recommend setting up automatic monthly "
                f"installments and keeping revolving credit card balances below 25% of your limits."
            )

        else:
            headline = f"Application Status: Your loan request of ${loan_amnt:,.0f} has been DECLINED."

            if gate_violations:
                summary = (
                    f"We carefully evaluated your loan request against institutional credit standards. Your application "
                    f"could not be approved primarily due to mandatory underwriting policy guidelines: {gate_violations[0]}"
                )
            else:
                summary = (
                    f"We carefully evaluated your credit profile against our statistical risk model. Your application "
                    f"resulted in an approval confidence score of {prob*100:.1f}%, which falls below our required 55.0% threshold. "
                    f"The decision was driven primarily by your current debt load and credit profile indicators."
                )

            # Key driving factors for rejection
            factor_highlights = []
            for item in top_neg[:3]:
                factor_highlights.append({
                    "factor": item["friendly_name"],
                    "impact": f"-{item['impact_pct']}% adverse influence",
                    "explanation": item["plain_narrative"]
                })

            recommendation = (
                "While we cannot approve your loan at this time, credit eligibility is dynamic. We have generated a "
                "customized action plan below with exact targets that can help you qualify upon re-application."
            )

        return {
            "headline": headline,
            "summary": summary,
            "primary_driving_factors": factor_highlights,
            "next_steps_advice": recommendation,
            "counterfactual_targets": explanation["counterfactuals"]
        }
