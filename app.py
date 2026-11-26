"""
Loan Eligibility Agent - Interactive Streamlit Application.
Provides interactive underwriting decisions, plain-language factor explanations,
factor contribution waterfall charts, dynamic what-if simulation, sensitivity curves,
conversational AI underwriter, and official notice generation.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.agent import LoanEligibilityAgent
from src.sample_decisions import SAMPLE_PERSONAS
from src.model import get_trained_model
from src.conversational_agent import ConversationalUnderwriter
from src.document_generator import generate_decision_letter_html
from src.data_pipeline import FEATURE_NAMES_FRIENDLY

# Set page config
st.set_page_config(
    page_title="LendingClub Loan Eligibility Agent",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .decision-approved {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2);
    }
    .decision-rejected {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.2);
    }
    .factor-card-pos {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .factor-card-neg {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_agent():
    model = get_trained_model()
    agent = LoanEligibilityAgent(model=model)
    conv_agent = ConversationalUnderwriter(agent=agent)
    return agent, model, conv_agent


agent, model, conv_agent = load_agent()

# Header
st.markdown('<div class="main-header">🏦 LendingClub Loan Eligibility Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Autonomous Credit Underwriting Engine with Explainable AI & Plain-Language Factor Attribution (Trained on Kaggle LendingClub Data)</div>',
    unsafe_allow_html=True
)

# Sidebar: LLM Model Configuration
st.sidebar.markdown("### 🤖 LLM Model Settings")
with st.sidebar.expander("LLM Provider & Model Setup", expanded=True):
    llm_provider = st.selectbox(
        "Select LLM Provider",
        ["Google Gemini", "Groq (100% Free Tier)", "OpenAI"],
        index=0
    )

    if llm_provider == "Google Gemini":
        api_key_default = os.environ.get("GEMINI_API_KEY", "")
        model_options = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
        provider_key = "google"
        st.caption("🔑 Free key: [Google AI Studio](https://aistudio.google.com/app/apikey)")
        with st.expander("⚠️ 'Can't create key' in AI Studio?"):
            st.markdown("""
            1. **Use Personal Gmail:** If logged in with a college/university email, Google blocks API key creation. Open an **Incognito window** and sign in with a personal `@gmail.com`.
            2. **New Project:** When prompted, choose **'Create API key in new project'**.
            3. **Or Switch to Groq:** Select **Groq (100% Free Tier)** above — keys generate in 10s without Google Cloud restrictions!
            """)
    elif llm_provider == "Groq (100% Free Tier)":
        api_key_default = os.environ.get("GROQ_API_KEY", "")
        model_options = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        provider_key = "groq"
        st.success("⚡ **Instant Free Key:** Get your free key at [console.groq.com/keys](https://console.groq.com/keys) (Takes 10s, no credit card).")
    else:
        api_key_default = os.environ.get("OPENAI_API_KEY", "")
        model_options = ["gpt-4o-mini", "gpt-4o"]
        provider_key = "openai"
        st.caption("🔑 Key from: [OpenAI Platform](https://platform.openai.com/api-keys)")

    user_api_key = st.text_input(
        f"{llm_provider} API Key",
        value=api_key_default,
        type="password",
        help="Enter your API key or set environment variable."
    )
    model_choice = st.selectbox(
        "Model Variant",
        model_options,
        index=0
    )

    if hasattr(agent, "llm_engine") and agent.llm_engine:
        if user_api_key:
            agent.llm_engine.configure(api_key=user_api_key, model_name=model_choice, provider=provider_key)
        elif agent.llm_engine.provider != provider_key or agent.llm_engine.model_name != model_choice:
            agent.llm_engine.configure(model_name=model_choice, provider=provider_key)

        llm_status = agent.llm_engine.get_status()
        if llm_status["active"]:
            st.success(f"🟢 Active: `{llm_status['provider']}` ({llm_status['model_name']})")
        else:
            st.info("🟡 Offline Heuristic Mode (Enter API Key above to activate LLM)")
        use_llm_toggle = st.toggle("Enable LLM Synthesis", value=llm_status["active"])
    else:
        use_llm_toggle = False
        st.warning("LLM Engine not initialized")

st.sidebar.divider()

# Sidebar: Persona Quick Loader & Manual Inputs
st.sidebar.header("📋 Applicant Profile")

preset_options = {
    "Custom Input": None,
    "Sarah Jenkins (Prime Approved - Grade A)": "approved_prime",
    "Marcus Vance (Near-Prime Approved - Grade C)": "approved_near_prime",
    "David Miller (High Debt Overextended - Rejected)": "rejected_high_debt",
    "Elena Rostova (Policy & Credit History - Rejected)": "rejected_adverse_history",
    "Carlos Gomez (Borderline Counterfactual - Rejected)": "borderline_counterfactual"
}

selected_preset_label = st.sidebar.selectbox(
    "⚡ Load Sample Decision Persona:",
    list(preset_options.keys()),
    index=1
)

selected_preset_key = preset_options[selected_preset_label]
defaults = SAMPLE_PERSONAS.get(selected_preset_key, SAMPLE_PERSONAS["approved_prime"]) if selected_preset_key else {
    "applicant_id": "APP-CUSTOM",
    "name": "Custom Applicant",
    "annual_inc": 75000.0,
    "loan_amnt": 18000.0,
    "term": "36 months",
    "fico_score": 710.0,
    "dti": 18.5,
    "emp_length_years": 5.0,
    "home_ownership": "MORTGAGE",
    "purpose": "debt_consolidation",
    "delinq_2yrs": 0,
    "inq_last_6mths": 1,
    "open_acc": 12,
    "total_acc": 24,
    "revol_bal": 8500.0,
    "revol_util": 35.0,
    "pub_rec_bankruptcies": 0
}

# Form inputs
with st.sidebar.expander("💵 Loan & Income Details", expanded=True):
    loan_amnt = st.number_input(
        "Requested Loan Amount ($)",
        min_value=1000,
        max_value=40000,
        value=int(defaults["loan_amnt"]),
        step=500
    )
    annual_inc = st.number_input(
        "Annual Income ($)",
        min_value=5000,
        max_value=500000,
        value=int(defaults["annual_inc"]),
        step=1000
    )
    term = st.selectbox(
        "Loan Term",
        ["36 months", "60 months"],
        index=0 if defaults["term"] == "36 months" else 1
    )
    purpose_list = [
        "debt_consolidation", "credit_card", "home_improvement",
        "major_purchase", "small_business", "car", "medical", "other"
    ]
    purpose = st.selectbox(
        "Loan Purpose",
        purpose_list,
        index=purpose_list.index(defaults["purpose"]) if defaults["purpose"] in purpose_list else 0
    )

with st.sidebar.expander("📊 Credit History & Debt", expanded=True):
    fico_score = st.slider(
        "FICO Credit Score",
        min_value=500,
        max_value=850,
        value=int(defaults["fico_score"]),
        step=5
    )
    dti = st.slider(
        "Debt-to-Income (DTI %)",
        min_value=1.0,
        max_value=55.0,
        value=float(defaults["dti"]),
        step=0.5
    )
    revol_util = st.slider(
        "Revolving Credit Utilization (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(defaults["revol_util"]),
        step=1.0
    )
    revol_bal = st.number_input(
        "Revolving Balance ($)",
        min_value=0,
        max_value=150000,
        value=int(defaults["revol_bal"]),
        step=500
    )

with st.sidebar.expander("👤 Background & Derogatory Marks", expanded=False):
    emp_length_years = st.slider(
        "Employment Length (Years)",
        min_value=0.0,
        max_value=15.0,
        value=float(defaults["emp_length_years"]),
        step=1.0
    )
    home_ownership = st.selectbox(
        "Home Ownership",
        ["MORTGAGE", "RENT", "OWN"],
        index=["MORTGAGE", "RENT", "OWN"].index(defaults["home_ownership"]) if defaults["home_ownership"] in ["MORTGAGE", "RENT", "OWN"] else 0
    )
    delinq_2yrs = st.number_input(
        "Delinquencies (Past 2 Yrs)",
        min_value=0,
        max_value=10,
        value=int(defaults["delinq_2yrs"]),
        step=1
    )
    inq_last_6mths = st.number_input(
        "Inquiries (Last 6 Mos)",
        min_value=0,
        max_value=10,
        value=int(defaults["inq_last_6mths"]),
        step=1
    )
    open_acc = st.number_input(
        "Open Credit Lines",
        min_value=1,
        max_value=50,
        value=int(defaults["open_acc"]),
        step=1
    )
    total_acc = st.number_input(
        "Total Credit Lines",
        min_value=1,
        max_value=70,
        value=int(defaults["total_acc"]),
        step=1
    )
    pub_rec_bankruptcies = st.number_input(
        "Public Record Bankruptcies",
        min_value=0,
        max_value=5,
        value=int(defaults["pub_rec_bankruptcies"]),
        step=1
    )

# Build current profile
current_profile = {
    "applicant_id": defaults.get("applicant_id", "APP-CUSTOM"),
    "loan_amnt": float(loan_amnt),
    "annual_inc": float(annual_inc),
    "term": term,
    "purpose": purpose,
    "fico_score": float(fico_score),
    "dti": float(dti),
    "revol_util": float(revol_util),
    "revol_bal": float(revol_bal),
    "emp_length_years": float(emp_length_years),
    "home_ownership": home_ownership,
    "delinq_2yrs": int(delinq_2yrs),
    "inq_last_6mths": int(inq_last_6mths),
    "open_acc": int(open_acc),
    "total_acc": int(total_acc),
    "pub_rec_bankruptcies": int(pub_rec_bankruptcies)
}

# Run Underwriting Agent
result = agent.evaluate_applicant(current_profile, use_llm=use_llm_toggle)

# Display Decision Banner
is_approved = result["decision"] == "APPROVED"
prob_pct = result["approval_probability"] * 100.0

col_decision, col_metrics = st.columns([1.2, 2.0])

with col_decision:
    if is_approved:
        st.markdown(f"""
        <div class="decision-approved">
            <h2 style="margin: 0; font-size: 2rem;">✅ APPROVED</h2>
            <p style="margin-top: 8px; font-size: 1.1rem; opacity: 0.95;">
                {result['plain_language_explanation']['headline']}
            </p>
            <div style="margin-top: 15px; font-weight: bold; background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 8px; display: inline-block;">
                Approval Confidence: {prob_pct:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="decision-rejected">
            <h2 style="margin: 0; font-size: 2rem;">❌ DECLINED</h2>
            <p style="margin-top: 8px; font-size: 1.1rem; opacity: 0.95;">
                {result['plain_language_explanation']['headline']}
            </p>
            <div style="margin-top: 15px; font-weight: bold; background: rgba(255,255,255,0.2); padding: 8px 16px; border-radius: 8px; display: inline-block;">
                Approval Confidence: {prob_pct:.1f}% (Required: ≥55.0%)
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_metrics:
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="LendingClub Grade",
            value=f"Grade {result['grade']}",
            delta="Prime Tier" if result['grade'] in ['A', 'B'] else "Higher Risk"
        )
    with m2:
        st.metric(
            label="Estimated APR",
            value=f"{result['estimated_apr']:.2f}%",
            delta="- Low Rate" if result['estimated_apr'] < 12.0 else "+ Risk Premium"
        )
    with m3:
        n_m = 60 if term == "60 months" else 36
        r = (result['estimated_apr'] / 100.0) / 12.0
        installment = loan_amnt * (r * (1 + r)**n_m) / ((1 + r)**n_m - 1)
        st.metric(
            label="Est. Monthly Payment",
            value=f"${installment:,.2f}",
            help=f"Based on {term} amortization at {result['estimated_apr']:.2f}% APR"
        )
    with m4:
        st.metric(
            label="Credit Bureau Score",
            value=f"{result['risk_score']}",
            delta=f"{result['risk_score'] - 690:+d} vs LC Avg"
        )

    st.write(f"**Underwriting Risk Score Gauge:** {prob_pct:.1f}% of benchmark criteria satisfied")
    st.progress(min(1.0, max(0.0, result["approval_probability"])))

st.divider()

# Main Tabs
tab_explain, tab_waterfall, tab_counterfactual, tab_chat, tab_adverse, tab_letter, tab_dataset = st.tabs([
    "🗣️ Plain-Language Explanation",
    "📊 Factor Attribution (Exact Drivers)",
    "🎯 What-If & Sensitivity Curves",
    "💬 Ask Underwriter Agent (Q&A)",
    "📄 Adverse Action Notice (Compliance)",
    "✉️ Official Decision Notice",
    "📈 Dataset & Model Insights"
])

with tab_explain:
    st.subheader("💡 Underwriting Agent Plain-Language Briefing")
    
    if result.get("llm_enhanced"):
        st.success(f"✨ **Synthesized by Google Gemini ({result.get('llm_model')})** with exact closed-form factor attributions.")
        llm_narrative = result['plain_language_explanation'].get('llm_narrative')
        if llm_narrative:
            st.markdown("#### 🏛️ LLM Senior Credit Officer Commentary")
            st.markdown(llm_narrative)
            st.markdown("---")
    else:
        st.markdown(f"**Executive Summary:** {result['plain_language_explanation']['summary']}")
        if hasattr(agent, "llm_engine") and not agent.llm_engine.is_active:
            st.caption("ℹ️ *To generate full LLM-synthesized underwriting narratives, provide a Gemini API Key in the sidebar.*")

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("### 🟢 Top Positive Drivers (Boosting Approval)")
        pos_factors = result["factor_attribution"]["top_positive_factors"]
        if pos_factors:
            for f in pos_factors[:4]:
                st.markdown(f"""
                <div class="factor-card-pos">
                    <strong style="color: #065F46;">{f['friendly_name']}</strong>
                    <span style="float: right; color: #059669; font-weight: bold;">+{f['impact_pct']}% impact</span><br>
                    <span style="font-size: 0.95rem; color: #1F2937;">{f['plain_narrative']}</span><br>
                    <small style="color: #6B7280;">Applicant Value: <b>{f['applicant_value']}</b> | Benchmark: {f['benchmark_value'] if isinstance(f['benchmark_value'], str) else round(f['benchmark_value'], 1)}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No significant positive factors identified.")

    with col_neg:
        st.markdown("### 🔴 Top Negative Drivers (Risk Factors)")
        neg_factors = result["factor_attribution"]["top_negative_factors"]
        if neg_factors:
            for f in neg_factors[:4]:
                st.markdown(f"""
                <div class="factor-card-neg">
                    <strong style="color: #991B1B;">{f['friendly_name']}</strong>
                    <span style="float: right; color: #DC2626; font-weight: bold;">-{f['impact_pct']}% impact</span><br>
                    <span style="font-size: 0.95rem; color: #1F2937;">{f['plain_narrative']}</span><br>
                    <small style="color: #6B7280;">Applicant Value: <b>{f['applicant_value']}</b> | Benchmark: {f['benchmark_value'] if isinstance(f['benchmark_value'], str) else round(f['benchmark_value'], 1)}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No significant negative risk factors identified.")

    st.markdown("---")
    st.markdown(f"**Advisor Recommendation & Next Steps:** {result['plain_language_explanation']['next_steps_advice']}")

with tab_waterfall:
    st.subheader("🔍 Mathematical Factor Contribution Decomposition")
    st.write(
        "Each factor's log-odds contribution ($w_i \\cdot x_i$) relative to the LendingClub population benchmark. "
        "Positive values push towards **Approval**, negative values push towards **Decline**."
    )

    all_factors = result["factor_attribution"]["all_factors"]
    chart_df = pd.DataFrame([
        {
            "Factor": f["friendly_name"],
            "Contribution (Log-Odds)": f["log_odds_impact"],
            "Impact (%)": f["impact_pct"],
            "Type": "Favorable (Boosts Approval)" if f["log_odds_impact"] > 0 else "Adverse (Risk Driver)"
        }
        for f in all_factors
    ]).sort_values(by="Contribution (Log-Odds)", ascending=True)

    fig = px.bar(
        chart_df,
        x="Contribution (Log-Odds)",
        y="Factor",
        orientation="h",
        color="Type",
        color_discrete_map={
            "Favorable (Boosts Approval)": "#10B981",
            "Adverse (Risk Driver)": "#EF4444"
        },
        title=f"Factor Contributions to Decision for {current_profile.get('applicant_id', 'Applicant')}",
        text=chart_df["Impact (%)"].apply(lambda x: f"{x:.1f}%")
    )
    fig.update_layout(
        xaxis_title="Log-Odds Impact (Relative to Population Benchmark)",
        yaxis_title="",
        height=520,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_counterfactual:
    st.subheader("🎯 Counterfactual Guidance & Real-Time Sensitivity Curves")
    st.write(
        "Credit underwriting is dynamic. The charts below demonstrate the exact mathematical "
        "inflection points where this applicant crosses into approval or upgrades their credit tier."
    )

    cfs = result["plain_language_explanation"]["counterfactual_targets"]
    if cfs:
        for idx, cf in enumerate(cfs):
            st.markdown(f"""
            #### 📌 Milestone {idx+1}: {cf['action']}
            * **Current Profile:** `{cf['current']}` ➔ **Target Milestone:** `{cf['target']}`
            * **Expected Impact:** {cf['impact']}
            * **Actionable Step:** {cf['how_to_achieve']}
            """)
            st.divider()
    else:
        st.success("This applicant already qualifies for prime Grade A credit! No counterfactual remediation required.")

    st.markdown("### 📈 Interactive Sensitivity Simulation Curves")
    col_curve1, col_curve2 = st.columns(2)

    with col_curve1:
        # Curve 1: Sensitivity to Loan Amount
        loan_candidates = np.linspace(2000, 40000, 25)
        probs_by_loan = []
        for l_val in loan_candidates:
            p_copy = dict(current_profile)
            p_copy["loan_amnt"] = float(l_val)
            s_res = model.predict_score(p_copy)
            probs_by_loan.append(s_res["approval_probability"] * 100)

        fig_loan = go.Figure()
        fig_loan.add_trace(go.Scatter(
            x=loan_candidates,
            y=probs_by_loan,
            mode='lines+markers',
            name='Approval Probability',
            line=dict(color='#3B82F6', width=3)
        ))
        fig_loan.add_hline(y=55.0, line_dash="dash", line_color="#EF4444", annotation_text="55% Approval Threshold")
        fig_loan.add_vline(x=loan_amnt, line_dash="dot", line_color="#10B981", annotation_text=f"Current: ${loan_amnt:,.0f}")
        fig_loan.update_layout(
            title="Approval Probability vs Requested Loan Amount",
            xaxis_title="Requested Loan Amount ($)",
            yaxis_title="Approval Probability (%)",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_loan, use_container_width=True)

    with col_curve2:
        # Curve 2: Sensitivity to FICO Score
        fico_candidates = np.linspace(580, 850, 25)
        probs_by_fico = []
        for f_val in fico_candidates:
            p_copy = dict(current_profile)
            p_copy["fico_score"] = float(f_val)
            s_res = model.predict_score(p_copy)
            probs_by_fico.append(s_res["approval_probability"] * 100)

        fig_fico = go.Figure()
        fig_fico.add_trace(go.Scatter(
            x=fico_candidates,
            y=probs_by_fico,
            mode='lines+markers',
            name='Approval Probability',
            line=dict(color='#8B5CF6', width=3)
        ))
        fig_fico.add_hline(y=55.0, line_dash="dash", line_color="#EF4444", annotation_text="55% Approval Threshold")
        fig_fico.add_vline(x=fico_score, line_dash="dot", line_color="#10B981", annotation_text=f"Current: {int(fico_score)}")
        fig_fico.update_layout(
            title="Approval Probability vs FICO Credit Score",
            xaxis_title="FICO Score (300 - 850)",
            yaxis_title="Approval Probability (%)",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_fico, use_container_width=True)

with tab_chat:
    st.subheader("💬 Ask the Autonomous Underwriting Agent")
    st.write(
        "Have questions about this loan decision? Type below to chat with the agent or simulate dynamic what-if scenarios."
    )
    if hasattr(agent, "llm_engine") and agent.llm_engine.is_active and use_llm_toggle:
        st.caption(f"🟢 **LLM Active:** Conversing via Google Gemini (`{agent.llm_engine.model_name}`) grounded on applicant credit dossier.")
    else:
        st.caption("⚡ **Mode:** Heuristic Underwriter Simulation. Enter a Gemini API Key in the sidebar to enable open-ended LLM chat.")

    # Chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("agent", f"Hello! I am your AI Underwriting Assistant for {current_profile.get('applicant_id')}. How can I assist you with this decision?")
        ]

    user_query = st.text_input(
        "Ask a question or test a scenario (e.g. 'Why was I declined?', 'What if I borrow $10,000?', 'What if my FICO was 720?'):",
        key="chat_input"
    )

    if st.button("Submit Question", type="primary") and user_query:
        # Get agent response
        formatted_history = [{"role": s, "content": m} for s, m in st.session_state.chat_history]
        answer = conv_agent.respond(
            query=user_query,
            applicant_profile=current_profile,
            evaluation_result=result,
            chat_history=formatted_history,
            use_llm=use_llm_toggle
        )
        st.session_state.chat_history.append(("user", user_query))
        st.session_state.chat_history.append(("agent", answer))

    for sender, msg in reversed(st.session_state.chat_history):
        if sender == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").markdown(msg)

with tab_adverse:
    st.subheader("📄 Formal Adverse Action Notice (FCRA & ECOA Regulation B)")
    st.write(
        "In compliance with the Equal Credit Opportunity Act (ECOA) and the Fair Credit Reporting Act (FCRA), "
        "here are the principal reasons that adversely affected the credit decision in strict order of mathematical influence:"
    )

    reasons = result["factor_attribution"]["adverse_action_reasons"]
    if reasons:
        adverse_table = pd.DataFrame([
            {
                "Rank": r["rank"],
                "Principal Adverse Factor": r["factor"],
                "Applicant Value": r["value"],
                "Impact Weight": r["impact"],
                "Consumer Disclosure Reason": r["explanation"]
            }
            for r in reasons
        ])
        st.table(adverse_table)
    else:
        st.info("No adverse action reasons: Applicant was approved under prime criteria.")

    if result.get("policy_violations"):
        st.error(f"**Policy Gate Violation Notice:** {result['policy_violations'][0]}")

with tab_letter:
    st.subheader("✉️ Official Bank Decision Letter & Notice")
    st.write("Generate and download a formal, printable credit notice for this applicant.")

    html_letter = generate_decision_letter_html(result)

    st.download_button(
        label="📥 Download Official Decision Letter (HTML / Printable)",
        data=html_letter,
        file_name=f"Loan_Decision_{result['applicant_id'].split()[0]}.html",
        mime="text/html"
    )

    st.components.v1.html(html_letter, height=650, scrolling=True)

with tab_dataset:
    st.subheader("📊 LendingClub Dataset & Model Performance")
    m_info = model.metrics
    st.markdown(f"""
    * **Dataset Source:** LendingClub Loan Data (Kaggle)
    * **Training Size:** {m_info.get('n_train', 20000):,} loans
    * **Testing Split:** {m_info.get('n_test', 5000):,} loans
    * **ROC-AUC Score:** `{m_info.get('roc_auc', 0.89):.4f}`
    * **Underwriting Accuracy:** `{m_info.get('accuracy', 0.79):.2%}`
    * **Approval Precision:** `{m_info.get('precision', 0.97):.2%}`
    * **Approval Recall:** `{m_info.get('recall', 0.78):.2%}`
    """)

    cm = m_info.get("confusion_matrix", [[571, 103], [962, 3364]])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Charged Off (0)", "Actual Fully Paid (1)"],
        columns=["Predicted Declined (0)", "Predicted Approved (1)"]
    )
    st.write("**Confusion Matrix on Held-Out Test Set:**")
    st.dataframe(cm_df)

# Footer
st.markdown("---")
st.caption(
    "LendingClub Loan Eligibility Agent | Built with Scikit-Learn, Streamlit, FastAPI, and Explainable AI (XAI) | "
    "Features strictly conform to Kaggle LendingClub schema and Federal Credit Underwriting compliance standards."
)
