"""
Unit and Integration Tests for Loan Eligibility Agent.
Verifies dataset loading, model training, policy enforcement, explainability, and persona evaluation.
"""

import unittest
import os
import pandas as pd
from src.agent import LoanEligibilityAgent, UnderwritingPolicy
from src.model import LoanEligibilityModel, MODEL_PATH
from src.data_pipeline import get_train_test_data, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.sample_decisions import SAMPLE_PERSONAS


class TestLoanEligibilityAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.agent = LoanEligibilityAgent()
        cls.model = cls.agent.model

    def test_data_pipeline(self):
        """Verify train-test split integrity and features."""
        X_train, X_test, y_train, y_test = get_train_test_data()
        self.assertGreater(len(X_train), 1000)
        self.assertGreater(len(X_test), 200)
        for col in NUMERICAL_FEATURES:
            self.assertIn(col, X_train.columns)
        for col in CATEGORICAL_FEATURES:
            self.assertIn(col, X_train.columns)

    def test_model_metrics(self):
        """Verify model performance on test set meets credit industry standard (ROC-AUC > 0.80)."""
        metrics = self.model.metrics
        self.assertIn("roc_auc", metrics)
        self.assertGreater(metrics["roc_auc"], 0.80)
        self.assertGreater(metrics["accuracy"], 0.70)

    def test_policy_gates(self):
        """Verify hard policy violations trigger appropriate decline flags."""
        bad_fico_profile = {"fico_score": 520, "dti": 15.0, "annual_inc": 50000, "pub_rec_bankruptcies": 0}
        violations = UnderwritingPolicy.evaluate_gates(bad_fico_profile)
        self.assertTrue(any("580" in v for v in violations))

        bad_dti_profile = {"fico_score": 700, "dti": 52.0, "annual_inc": 50000, "pub_rec_bankruptcies": 0}
        violations = UnderwritingPolicy.evaluate_gates(bad_dti_profile)
        self.assertTrue(any("Debt-to-Income" in v for v in violations))

    def test_approved_applicant(self):
        """Verify prime persona results in Approval with positive driver attribution."""
        prime = SAMPLE_PERSONAS["approved_prime"]
        result = self.agent.evaluate_applicant(prime)

        self.assertEqual(result["decision"], "APPROVED")
        self.assertTrue(result["is_eligible"])
        self.assertGreater(result["approval_probability"], 0.70)
        self.assertIn(result["grade"], ["A", "B"])

        exp = result["plain_language_explanation"]
        self.assertIn("APPROVED", exp["headline"])
        self.assertGreater(len(exp["primary_driving_factors"]), 0)

        # Check positive factors contain credit score
        pos_names = [f["factor"] for f in exp["primary_driving_factors"]]
        self.assertTrue(any("Credit Score" in name for name in pos_names))

    def test_rejected_applicant(self):
        """Verify overextended persona results in Decline with adverse driver attribution."""
        rejected = SAMPLE_PERSONAS["rejected_high_debt"]
        result = self.agent.evaluate_applicant(rejected)

        self.assertEqual(result["decision"], "REJECTED")
        self.assertFalse(result["is_eligible"])
        self.assertLess(result["approval_probability"], 0.55)

        exp = result["plain_language_explanation"]
        self.assertIn("DECLINED", exp["headline"])
        self.assertGreater(len(exp["primary_driving_factors"]), 0)
        self.assertGreater(len(exp["counterfactual_targets"]), 0)

    def test_factor_attribution_integrity(self):
        """Verify factor percentages sum towards ~100% and have consistent signs."""
        persona = SAMPLE_PERSONAS["borderline_counterfactual"]
        result = self.agent.evaluate_applicant(persona)

        factors = result["factor_attribution"]["all_factors"]
        total_pct = sum(f["impact_pct"] for f in factors)
        self.assertAlmostEqual(total_pct, 100.0, delta=2.0)

    def test_conversational_agent_what_if(self):
        """Verify conversational assistant dynamically simulates loan amount adjustments."""
        from src.conversational_agent import ConversationalUnderwriter
        conv = ConversationalUnderwriter(agent=self.agent)
        persona = SAMPLE_PERSONAS["rejected_high_debt"]
        eval_res = self.agent.evaluate_applicant(persona)

        # Test what if loan amount reduction
        query = "What if I only ask for $10,000?"
        response = conv.respond(query, persona, eval_res)
        self.assertIn("Simulation Result", response)
        self.assertIn("$10,000", response)

        # Test explanation of rejection
        response_why = conv.respond("Why was I rejected?", persona, eval_res)
        self.assertIn("Why Was This Loan Declined", response_why)

    def test_document_generator(self):
        """Verify HTML decision notice generation."""
        from src.document_generator import generate_decision_letter_html
        persona = SAMPLE_PERSONAS["approved_prime"]
        eval_res = self.agent.evaluate_applicant(persona)
        html = generate_decision_letter_html(eval_res)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("APPROVED", html)
        self.assertIn("Sarah Jenkins", html)

    def test_gemini_engine_fallback(self):
        """Verify GeminiUnderwritingEngine falls back cleanly when no key is configured."""
        from src.llm_engine import GeminiUnderwritingEngine
        engine = GeminiUnderwritingEngine(api_key=None)
        status = engine.get_status()
        self.assertFalse(status["active"])
        self.assertEqual(status["provider"], "Google Gemini")

        persona = SAMPLE_PERSONAS["approved_prime"]
        eval_res = self.agent.evaluate_applicant(persona, use_llm=True)
        self.assertIn("plain_language_explanation", eval_res)
        self.assertIn("APPROVED", eval_res["plain_language_explanation"]["headline"])

    def test_gemini_prompt_formatting(self):
        """Verify underwriting prompt construction with exact factor attributions."""
        from src.llm_engine import GeminiUnderwritingEngine
        engine = GeminiUnderwritingEngine(api_key=None)
        persona = SAMPLE_PERSONAS["rejected_high_debt"]
        eval_res = self.agent.evaluate_applicant(persona, use_llm=False)
        prompt = engine._build_underwriting_prompt(
            profile=eval_res["raw_profile"],
            decision=eval_res["decision"],
            decision_type=eval_res["decision_type"],
            gate_violations=eval_res["policy_violations"],
            scoring={
                "approval_probability": eval_res["approval_probability"],
                "credit_score": eval_res["risk_score"],
                "grade": eval_res["grade"],
                "estimated_apr": eval_res["estimated_apr"]
            },
            explanation=eval_res["factor_attribution"]
        )
        self.assertIn("LendingClub", prompt)
        self.assertIn("REJECTED", prompt)
        self.assertIn("Debt-to-Income", prompt)
        self.assertIn("COUNTERFACTUAL", prompt)


if __name__ == "__main__":
    unittest.main()
