"""
Universal LLM Engine for Loan Eligibility Underwriting & Explainability.
Supports Google Gemini, Groq (100% Free Tier with Llama 3.3/3.1), and OpenAI.
Provides LLM-driven synthesis of mathematical factor attributions,
plain-language credit underwriting dossiers, and conversational Q&A.
"""

import os
import json
import re
import logging
import warnings
from typing import Dict, Any, List, Optional, Callable

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


def _load_env_file():
    """Lightweight loader for .env file if present in workspace root."""
    try:
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
    except Exception as e:
        logger.debug(f"Could not load .env file: {e}")


# Auto-load .env on import
_load_env_file()


class GeminiUnderwritingEngine:
    """
    LLM Underwriting Engine supporting Google Gemini, Groq (Free), and OpenAI.
    Converts quantitative underwriting scorecards and exact log-odds XAI
    attributions into natural language explanations and interactive assistance.
    """

    DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
    DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
    DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: str = "google"
    ):
        self.provider = provider.lower()
        self.model_name = model_name or self.DEFAULT_GEMINI_MODEL
        self.api_key: Optional[str] = None
        self.client: Optional[Any] = None
        self.is_active: bool = False
        self._init_error: Optional[str] = None

        self.configure(api_key=api_key, model_name=model_name, provider=provider)

    def configure(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None
    ):
        """Configure or reconfigure the LLM provider and API client."""
        if provider:
            self.provider = provider.lower()

        # Auto-detect provider and key if not explicitly set
        if not api_key:
            if self.provider == "google" and os.environ.get("GEMINI_API_KEY"):
                key = os.environ.get("GEMINI_API_KEY")
            elif self.provider == "groq" and os.environ.get("GROQ_API_KEY"):
                key = os.environ.get("GROQ_API_KEY")
            elif self.provider == "openai" and os.environ.get("OPENAI_API_KEY"):
                key = os.environ.get("OPENAI_API_KEY")
            elif os.environ.get("GEMINI_API_KEY"):
                self.provider = "google"
                key = os.environ.get("GEMINI_API_KEY")
            elif os.environ.get("GROQ_API_KEY"):
                self.provider = "groq"
                key = os.environ.get("GROQ_API_KEY")
            elif os.environ.get("OPENAI_API_KEY"):
                self.provider = "openai"
                key = os.environ.get("OPENAI_API_KEY")
            else:
                key = None
        else:
            key = api_key

        # Set default model for the provider if none given
        if model_name:
            self.model_name = model_name
        elif self.provider == "groq" and ("gemini" in self.model_name or "gpt" in self.model_name):
            self.model_name = self.DEFAULT_GROQ_MODEL
        elif self.provider == "openai" and ("gemini" in self.model_name or "llama" in self.model_name):
            self.model_name = self.DEFAULT_OPENAI_MODEL
        elif self.provider == "google" and ("llama" in self.model_name or "gpt" in self.model_name):
            self.model_name = self.DEFAULT_GEMINI_MODEL

        if not key:
            self.api_key = None
            self.client = None
            self.is_active = False
            self._init_error = f"No API key provided for {self.provider.title()}"
            return

        self.api_key = key.strip()

        try:
            if self.provider == "google":
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model_name)
                self.is_active = True
                self._init_error = None
            elif self.provider == "groq":
                from openai import OpenAI
                self.client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=self.api_key
                )
                self.is_active = True
                self._init_error = None
            elif self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                self.is_active = True
                self._init_error = None
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
        except Exception as e:
            self.is_active = False
            self.client = None
            self._init_error = str(e)
            logger.warning(f"Failed to initialize {self.provider} model: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Return current status of LLM engine."""
        display_names = {
            "google": "Google Gemini",
            "groq": "Groq (Free)",
            "openai": "OpenAI"
        }
        return {
            "active": self.is_active,
            "provider": display_names.get(self.provider, self.provider.title()),
            "provider_key": self.provider,
            "model_name": self.model_name,
            "api_key_configured": bool(self.api_key),
            "init_error": self._init_error
        }

    def generate_underwriting_dossier(
        self,
        profile: Dict[str, Any],
        decision: str,
        decision_type: str,
        gate_violations: List[str],
        scoring: Dict[str, Any],
        explanation: Dict[str, Any],
        fallback_func: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Synthesize quantitative underwriting scorecard outputs and exact factor
        attributions into an institutional-grade, plain-language decision dossier.
        """
        # Baseline deterministic result
        baseline = fallback_func(
            profile=profile,
            decision=decision,
            decision_type=decision_type,
            gate_violations=gate_violations,
            scoring=scoring,
            explanation=explanation
        ) if fallback_func else {}

        if not self.is_active or not self.client:
            baseline["llm_enhanced"] = False
            baseline["llm_model"] = "none (heuristic fallback)"
            baseline["llm_narrative"] = baseline.get("summary", "")
            return baseline

        prompt = self._build_underwriting_prompt(
            profile=profile,
            decision=decision,
            decision_type=decision_type,
            gate_violations=gate_violations,
            scoring=scoring,
            explanation=explanation
        )

        try:
            if self.provider == "google":
                response = self.client.generate_content(
                    prompt,
                    generation_config={"temperature": 0.2, "max_output_tokens": 1024}
                )
                response_text = response.text.strip()
            else:
                # Groq or OpenAI
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1024
                )
                response_text = completion.choices[0].message.content.strip()

            parsed = self._parse_json_or_text(response_text)

            headline = parsed.get("headline", baseline.get("headline", f"Application Status: {decision}"))
            if decision == "APPROVED" and "APPROVED" not in headline.upper():
                headline = f"Congratulations! Your application for a ${profile.get('loan_amnt', 0):,.0f} loan has been APPROVED."
            elif decision == "REJECTED" and "DECLINED" not in headline.upper() and "REJECTED" not in headline.upper():
                headline = f"Application Status: Your loan request of ${profile.get('loan_amnt', 0):,.0f} has been DECLINED."

            summary = parsed.get("executive_summary", baseline.get("summary", ""))
            llm_narrative = parsed.get("llm_narrative", summary)

            primary_factors = parsed.get("primary_driving_factors")
            if not primary_factors or not isinstance(primary_factors, list):
                primary_factors = baseline.get("primary_driving_factors", [])

            next_steps = parsed.get("next_steps_advice", baseline.get("next_steps_advice", ""))

            return {
                "headline": headline,
                "summary": summary,
                "llm_narrative": llm_narrative,
                "primary_driving_factors": primary_factors,
                "next_steps_advice": next_steps,
                "counterfactual_targets": explanation.get("counterfactuals", baseline.get("counterfactual_targets", [])),
                "llm_enhanced": True,
                "llm_model": f"{self.model_name} ({self.provider.title()})"
            }

        except Exception as e:
            logger.warning(f"{self.provider.title()} API call failed, falling back to deterministic explanation: {e}")
            baseline["llm_enhanced"] = False
            baseline["llm_model"] = f"fallback ({str(e)})"
            baseline["llm_narrative"] = baseline.get("summary", "")
            return baseline

    def chat_underwriter(
        self,
        query: str,
        profile: Dict[str, Any],
        evaluation_result: Dict[str, Any],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Conduct interactive conversational credit underwriting Q&A using the active LLM.
        """
        if not self.is_active or not self.client:
            return ""

        context = self._format_applicant_context(profile, evaluation_result)
        system_instruction = (
            "You are an expert, empathetic Senior Credit Underwriter at LendingClub. "
            "You have full access to the applicant's financial profile, the machine learning credit scorecard "
            "metrics (probability, grade, APR), exact feature contributions (log-odds), policy gates, "
            "and counterfactual remediation milestones.\n\n"
            "RULES:\n"
            "1. Answer the applicant's or underwriter's question accurately based strictly on the provided dossier.\n"
            "2. If declined, explain the top mathematical adverse factors (e.g. FICO, DTI, revolving utilization) "
            "compassionately and reference concrete counterfactual numbers they can target.\n"
            "3. If asked about what-if changes (e.g. changing loan amount or FICO), give informed credit guidance.\n"
            "4. Keep explanations concise, professional, and free of financial jargon where possible.\n"
            "5. Comply with Equal Credit Opportunity Act (ECOA) and Regulation B requirements by never referencing "
            "unrelated protected attributes.\n\n"
            f"APPLICANT DOSSIER:\n{context}\n"
        )

        try:
            if self.provider == "google":
                history_text = ""
                if chat_history:
                    history_lines = []
                    for item in chat_history[-6:]:
                        role = "Applicant" if item.get("role") in ["user", "applicant"] else "Underwriter"
                        history_lines.append(f"{role}: {item.get('content', '')}")
                    history_text = "RECENT CONVERSATION HISTORY:\n" + "\n".join(history_lines) + "\n\n"

                prompt = f"{system_instruction}\n{history_text}Applicant: {query}\nUnderwriter:"
                response = self.client.generate_content(
                    prompt,
                    generation_config={"temperature": 0.3, "max_output_tokens": 512}
                )
                return response.text.strip()
            else:
                # Groq or OpenAI chat completions
                messages = [{"role": "system", "content": system_instruction}]
                if chat_history:
                    for item in chat_history[-6:]:
                        r = "user" if item.get("role") in ["user", "applicant"] else "assistant"
                        messages.append({"role": r, "content": item.get("content", "")})
                messages.append({"role": "user", "content": query})

                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=512
                )
                return completion.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"{self.provider.title()} conversational chat failed: {e}")
            return ""

    def _build_underwriting_prompt(
        self,
        profile: Dict[str, Any],
        decision: str,
        decision_type: str,
        gate_violations: List[str],
        scoring: Dict[str, Any],
        explanation: Dict[str, Any]
    ) -> str:
        """Construct structured prompt for LLM explanation generation."""
        top_pos = explanation.get("top_positive_factors", [])[:3]
        top_neg = explanation.get("top_negative_factors", [])[:3]
        counterfactuals = explanation.get("counterfactuals", [])

        pos_str = "\n".join([
            f"  - {f['friendly_name']} ({f.get('applicant_value', 'N/A')}): +{f.get('impact_pct', 0)}% favorable influence (baseline: {f.get('benchmark_value', 'N/A')})"
            for f in top_pos
        ]) or "  - None"

        neg_str = "\n".join([
            f"  - {f['friendly_name']} ({f.get('applicant_value', 'N/A')}): -{f.get('impact_pct', 0)}% adverse penalty (baseline: {f.get('benchmark_value', 'N/A')})"
            for f in top_neg
        ]) or "  - None"

        cf_str = "\n".join([
            f"  - {c.get('parameter', '')}: current {c.get('current_value', '')} -> target {c.get('target_milestone', '')} ({c.get('impact', '')})"
            for c in counterfactuals
        ]) or "  - None"

        violations_str = "\n".join([f"  - {v}" for v in gate_violations]) if gate_violations else "None"

        return f"""You are the Lead Credit Underwriting AI at LendingClub.
Analyze the following credit risk evaluation and synthesize an institutional, transparent, plain-language decision dossier adhering to FCRA / ECOA Regulation B guidelines.

APPLICANT PROFILE:
- Applicant ID: {profile.get('applicant_id', 'APP-UNKNOWN')}
- Gross Annual Income: ${profile.get('annual_inc', 0):,.2f}
- Requested Loan Amount: ${profile.get('loan_amnt', 0):,.2f} ({profile.get('term', '36 months')})
- Purpose: {profile.get('purpose', 'debt_consolidation')}
- FICO Credit Score: {profile.get('fico_score', 0)}
- Debt-to-Income (DTI): {profile.get('dti', 0):.1f}%
- Revolving Credit Utilization: {profile.get('revol_util', 0):.1f}%
- Recent Inquiries (6m): {profile.get('inq_last_6mths', 0)}
- Past 2yr Delinquencies: {profile.get('delinq_2yrs', 0)}
- Public Record Bankruptcies: {profile.get('pub_rec_bankruptcies', 0)}

UNDERWRITING DECISION & MODEL METRICS:
- Final Decision: {decision} ({decision_type})
- Approval Probability: {scoring.get('approval_probability', 0)*100:.1f}% (Policy Threshold: 55.0%)
- Calibrated Credit Risk Score: {scoring.get('credit_score', 'N/A')} / 850
- Assigned Risk Grade: Grade {scoring.get('grade', 'N/A')}
- Risk-Based Pricing (Estimated APR): {scoring.get('estimated_apr', 0):.2f}%
- Hard Policy Gate Violations: {violations_str}

EXACT XAI FACTOR CONTRIBUTIONS (Ranked by Mathematical Influence):
Top Favorable Factors:
{pos_str}

Top Adverse Factors:
{neg_str}

COUNTERFACTUAL REMEDIATION TARGETS:
{cf_str}

INSTRUCTIONS:
Return ONLY a valid JSON object with the following schema:
{{
  "headline": "A concise status headline containing the word 'APPROVED' or 'DECLINED'",
  "executive_summary": "A 2-3 sentence clear, high-level summary of the decision, approval probability, and grade",
  "llm_narrative": "A detailed 2-paragraph underwriting commentary synthesizing the mathematical drivers into a clear financial narrative for the borrower",
  "primary_driving_factors": [
    {{
      "factor": "Friendly factor name (e.g. 'Credit Score (FICO 775)')",
      "impact": "+XX% favorable influence OR -XX% adverse penalty",
      "explanation": "Clear explanation of how this specific metric influenced the decision"
    }}
  ],
  "next_steps_advice": "Actionable guidance or regulatory notice"
}}
"""

    def _format_applicant_context(self, profile: Dict[str, Any], evaluation_result: Dict[str, Any]) -> str:
        """Format applicant financial context for chat assistant."""
        fa = evaluation_result.get("factor_attribution", {})
        top_pos = [f"{f['friendly_name']} (+{f.get('impact_pct',0)}%)" for f in fa.get("top_positive_factors", [])[:3]]
        top_neg = [f"{f['friendly_name']} (-{f.get('impact_pct',0)}%)" for f in fa.get("top_negative_factors", [])[:3]]

        return (
            f"- Applicant ID: {profile.get('applicant_id', 'Applicant')}\n"
            f"- Income: ${profile.get('annual_inc', 0):,.0f} | Requested Loan: ${profile.get('loan_amnt', 0):,.0f} ({profile.get('term', '36 months')})\n"
            f"- FICO Score: {int(profile.get('fico_score', 0))} | DTI: {profile.get('dti', 0):.1f}% | Revolving Util: {profile.get('revol_util', 0):.1f}%\n"
            f"- Decision: {evaluation_result.get('decision', 'N/A')} ({evaluation_result.get('decision_type', 'N/A')})\n"
            f"- Approval Probability: {evaluation_result.get('approval_probability', 0)*100:.1f}%\n"
            f"- Credit Grade: Grade {evaluation_result.get('grade', 'N/A')} | Est. APR: {evaluation_result.get('estimated_apr', 0):.2f}%\n"
            f"- Top Positive Drivers: {', '.join(top_pos) if top_pos else 'None'}\n"
            f"- Top Adverse Drivers: {', '.join(top_neg) if top_neg else 'None'}\n"
            f"- Policy Gate Violations: {evaluation_result.get('policy_violations', [])}\n"
        )

    def _parse_json_or_text(self, text: str) -> Dict[str, Any]:
        """Safely parse JSON response from LLM, handling markdown code blocks."""
        clean_text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        clean_text = re.sub(r"\s*```$", "", clean_text, flags=re.MULTILINE).strip()

        try:
            return json.loads(clean_text)
        except Exception:
            match = re.search(r"(\{.*\})", clean_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except Exception:
                    pass

        return {
            "executive_summary": text[:300],
            "llm_narrative": text
        }
