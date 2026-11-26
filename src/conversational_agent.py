"""
Conversational AI Underwriting Assistant.
Provides conversational Q&A and interactive what-if simulation for loan applicants and credit officers.
"""

import re
from typing import Dict, Any, List, Optional
from src.agent import LoanEligibilityAgent


class ConversationalUnderwriter:
    """Conversational assistant that reasons over underwriting decisions and executes dynamic what-if simulations."""

    def __init__(self, agent: Optional[LoanEligibilityAgent] = None, llm_engine: Optional[Any] = None):
        self.agent = agent or LoanEligibilityAgent()
        self.llm_engine = llm_engine or getattr(self.agent, "llm_engine", None)

    def respond(
        self,
        query: str,
        applicant_profile: Dict[str, Any],
        evaluation_result: Dict[str, Any],
        chat_history: Optional[List[Dict[str, str]]] = None,
        use_llm: bool = True
    ) -> str:
        """
        Process user query against the applicant's profile and evaluation result.
        Dynamically runs simulation if a what-if adjustment is detected.
        Uses Google Gemini LLM for conversational reasoning when active.
        """
        q_lower = query.lower()

        # Check for What-If FICO adjustments: e.g. "what if my credit score was 720" or "fico was 700"
        fico_match = re.search(r"(?:fico|credit score|score)\D*?(\d{3})", q_lower)
        if ("what if" in q_lower or "if my" in q_lower or "if i" in q_lower) and fico_match:
            try:
                target_score = float(fico_match.group(1))
                if 500 <= target_score <= 850:
                    return self._simulate_fico_change(target_score, applicant_profile, evaluation_result)
            except Exception:
                pass

        # Check for What-If DTI adjustments: e.g. "what if my dti was 20%"
        dti_match = re.search(r"(?:dti|debt to income)\D*?(\d{1,2}(?:\.\d+)?)", q_lower)
        if ("what if" in q_lower or "if my" in q_lower or "if i" in q_lower or "lower" in q_lower) and dti_match:
            try:
                target_dti = float(dti_match.group(1))
                if 1.0 <= target_dti <= 60.0:
                    return self._simulate_dti_change(target_dti, applicant_profile, evaluation_result)
            except Exception:
                pass

        # Check for What-If Loan Amount adjustments: e.g. "what if I ask for 10000" or "$10k" or "$10,000"
        if any(w in q_lower for w in ["what if", "could i", "can i", "if i", "reduce", "borrow", "ask for"]):
            # Find numbers with optional $, commas, or k suffix
            amt_match = re.search(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\s*(k|thousand)?", q_lower)
            if amt_match:
                try:
                    raw_val = amt_match.group(1).replace(",", "")
                    amt = float(raw_val)
                    if amt_match.group(2) or amt < 100:
                        amt *= 1000
                    if 1000 <= amt <= 100000:
                        return self._simulate_loan_amount_change(amt, applicant_profile, evaluation_result)
                except Exception:
                    pass

        # If Gemini LLM is active, leverage generative reasoning for open questions
        if use_llm and self.llm_engine and getattr(self.llm_engine, "is_active", False):
            llm_reply = self.llm_engine.chat_underwriter(
                query=query,
                profile=applicant_profile,
                evaluation_result=evaluation_result,
                chat_history=chat_history
            )
            if llm_reply:
                return llm_reply

        # Fallback Heuristic Rule-Based Responses:
        # Check for "Why rejected" / "Why declined"
        if any(term in q_lower for term in ["why rejected", "why declined", "why was i rejected", "reason for decline", "why not approved"]):
            return self._explain_rejection_reasons(evaluation_result)

        # Check for "What is biggest factor" / "Top factor"
        if any(term in q_lower for term in ["biggest factor", "top factor", "main reason", "most important"]):
            return self._explain_top_factors(evaluation_result)

        # Check for "How to improve" / "How to get approved"
        if any(term in q_lower for term in ["how to improve", "how to get approved", "next steps", "remediation", "action plan"]):
            return self._explain_improvement_plan(evaluation_result)

        # Check for "What is LendingClub grade" / "What does Grade mean"
        if any(term in q_lower for term in ["grade", "subgrade", "apr"]):
            grade = evaluation_result.get("grade", "N/A")
            apr = evaluation_result.get("estimated_apr", 0)
            return (
                f"Your assigned LendingClub credit rating is **Grade {grade}**, which maps to an estimated interest rate "
                f"of **{apr:.2f}% APR**. LendingClub assigns grades from A (lowest risk, lowest rate starting at ~6%) "
                f"to G (highest risk, interest rates above 28%). Grades A and B represent prime low-risk borrowers, "
                f"while C and D represent near-prime, and E, F, G represent higher risk tiers."
            )

        # Check for "What is Adverse Action"
        if "adverse action" in q_lower or "regulation b" in q_lower or "fcra" in q_lower:
            return (
                "An **Adverse Action Notice** is a formal disclosure required by federal law (under the Equal Credit "
                "Opportunity Act - Regulation B and the Fair Credit Reporting Act). Whenever an applicant is denied credit "
                "or offered terms less favorable than standard prime rates, the lender is legally required to disclose the "
                "top 4 principal reasons that mathematically caused the adverse decision. Our agent provides these exact "
                "reasons with quantitative impact weights."
            )

        # Default conversational response
        return self._generate_contextual_overview(evaluation_result)

    def _simulate_loan_amount_change(self, new_amount: float, profile: Dict[str, Any], original_result: Dict[str, Any]) -> str:
        """Simulate change in loan amount requested."""
        sim_profile = dict(profile)
        sim_profile["loan_amnt"] = new_amount
        sim_result = self.agent.evaluate_applicant(sim_profile)

        orig_prob = original_result["approval_probability"] * 100
        new_prob = sim_result["approval_probability"] * 100
        orig_dec = original_result["decision"]
        new_dec = sim_result["decision"]

        status_change = ""
        if orig_dec != new_dec:
            status_change = f"🎉 **Decision flips from {orig_dec} to {new_dec}!**\n\n"

        return (
            f"### 🧪 Simulation Result: Requested Amount = ${new_amount:,.0f}\n\n"
            f"{status_change}"
            f"- **New Decision:** `{new_dec}` ({sim_result['decision_type']})\n"
            f"- **Approval Confidence:** `{orig_prob:.1f}%` ➔ **`{new_prob:.1f}%`**\n"
            f"- **Assigned Grade:** `Grade {sim_result['grade']}`\n"
            f"- **Estimated APR:** `{sim_result['estimated_apr']:.2f}%`\n\n"
            f"**Analysis:** Changing the requested loan from ${profile['loan_amnt']:,.0f} to ${new_amount:,.0f} "
            f"adjusts your loan-to-income ratio to {(new_amount/profile['annual_inc'])*100:.1f}%. "
            f"{'This brings your monthly debt service within comfortable underwriting limits.' if sim_result['is_eligible'] else 'However, debt load or credit history still requires further adjustment for full approval.'}"
        )

    def _simulate_fico_change(self, new_fico: float, profile: Dict[str, Any], original_result: Dict[str, Any]) -> str:
        """Simulate change in FICO credit score."""
        sim_profile = dict(profile)
        sim_profile["fico_score"] = new_fico
        sim_result = self.agent.evaluate_applicant(sim_profile)

        orig_prob = original_result["approval_probability"] * 100
        new_prob = sim_result["approval_probability"] * 100
        orig_apr = original_result["estimated_apr"]
        new_apr = sim_result["estimated_apr"]

        return (
            f"### 🧪 Simulation Result: FICO Score = {int(new_fico)}\n\n"
            f"- **New Decision:** `{sim_result['decision']}`\n"
            f"- **Approval Confidence:** `{orig_prob:.1f}%` ➔ **`{new_prob:.1f}%`**\n"
            f"- **New Credit Rating:** `Grade {sim_result['grade']}`\n"
            f"- **Estimated APR:** `{orig_apr:.2f}%` ➔ **`{new_apr:.2f}%`** (Rate savings: `{max(0.0, orig_apr - new_apr):.2f}%`)\n\n"
            f"**Analysis:** An improvement to a {int(new_fico)} credit score significantly strengthens your credit risk profile. "
            f"{'Your loan is now fully approved!' if sim_result['is_eligible'] and not original_result['is_eligible'] else 'This qualifies you for better pricing and lower monthly installments.'}"
        )

    def _simulate_dti_change(self, new_dti: float, profile: Dict[str, Any], original_result: Dict[str, Any]) -> str:
        """Simulate change in Debt-to-Income ratio."""
        sim_profile = dict(profile)
        sim_profile["dti"] = new_dti
        sim_result = self.agent.evaluate_applicant(sim_profile)

        orig_prob = original_result["approval_probability"] * 100
        new_prob = sim_result["approval_probability"] * 100

        return (
            f"### 🧪 Simulation Result: Debt-to-Income (DTI) = {new_dti:.1f}%\n\n"
            f"- **New Decision:** `{sim_result['decision']}`\n"
            f"- **Approval Confidence:** `{orig_prob:.1f}%` ➔ **`{new_prob:.1f}%`**\n"
            f"- **Assigned Grade:** `Grade {sim_result['grade']}`\n"
            f"- **Estimated APR:** `{sim_result['estimated_apr']:.2f}%`\n\n"
            f"**Analysis:** Lowering your DTI to {new_dti:.1f}% frees up disposable monthly income and reduces default risk."
        )

    def _explain_rejection_reasons(self, result: Dict[str, Any]) -> str:
        """Explain why the application was declined."""
        if result["decision"] == "APPROVED":
            return "Good news! This application was actually **APPROVED**, not declined. It satisfied all underwriting criteria."

        neg = result["plain_language_explanation"]["primary_driving_factors"]
        pv = result.get("policy_violations", [])

        lines = ["### 📋 Why Was This Loan Declined?\n"]
        if pv:
            lines.append(f"**Critical Policy Violation:** {pv[0]}\n")
        lines.append("The statistical risk scorecard declined this application due to these top adverse factors:\n")
        for f in neg:
            lines.append(f"- **{f['factor']}** ({f['impact']}): {f['explanation']}")

        return "\n".join(lines)

    def _explain_top_factors(self, result: Dict[str, Any]) -> str:
        """Explain the primary driving factors."""
        factors = result["factor_attribution"]["all_factors"][:3]
        lines = ["### 🔍 Top Influencing Factors on This Decision:\n"]
        for idx, f in enumerate(factors, 1):
            sign = "🟢 Favorable (+)" if f["direction"] == "positive" else "🔴 Adverse (-)"
            lines.append(f"{idx}. **{f['friendly_name']}** &mdash; {sign} ({f['impact_pct']}% impact)\n   {f['plain_narrative']}")
        return "\n".join(lines)

    def _explain_improvement_plan(self, result: Dict[str, Any]) -> str:
        """Explain actionable steps to get approved."""
        cfs = result["plain_language_explanation"]["counterfactual_targets"]
        if not cfs:
            return "Your profile already qualifies for prime approval! To keep your Grade A rating, maintain low credit card utilization (<20%) and set up autopay."

        lines = ["### 🎯 Action Plan to Qualify Upon Re-Application:\n"]
        for idx, cf in enumerate(cfs, 1):
            lines.append(
                f"{idx}. **{cf['action']}**\n"
                f"   - Current: `{cf['current']}` ➔ Target: `{cf['target']}`\n"
                f"   - Action: {cf['how_to_achieve']}\n"
            )
        return "\n".join(lines)

    def _generate_contextual_overview(self, result: Dict[str, Any]) -> str:
        """Generate high-level assistant greeting and options."""
        return (
            f"I am your autonomous Underwriting Assistant. For this applicant ({result['applicant_id']}):\n\n"
            f"- **Decision:** `{result['decision']}` ({result['approval_probability']*100:.1f}% confidence)\n"
            f"- **Rating:** `Grade {result['grade']}` ({result['estimated_apr']:.2f}% APR)\n\n"
            f"You can ask me questions such as:\n"
            f"- *'Why was this loan declined?'*\n"
            f"- *'What if I only borrow $10,000?'*\n"
            f"- *'What if my credit score was 720?'*\n"
            f"- *'What is the single biggest factor affecting this decision?'*\n"
            f"- *'How can I get approved next time?'*"
        )
