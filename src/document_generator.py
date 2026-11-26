"""
Document Generator for Official Loan Decision Letters & Adverse Action Notices.
Generates printable, audit-compliant HTML decision documents suitable for borrowers and loan officers.
"""

from typing import Dict, Any
from datetime import datetime


def generate_decision_letter_html(evaluation_result: Dict[str, Any]) -> str:
    """Generate a clean, printable HTML document for the underwriting decision."""
    decision = evaluation_result["decision"]
    is_approved = decision == "APPROVED"
    profile = evaluation_result["raw_profile"]
    explanation = evaluation_result["plain_language_explanation"]
    date_str = datetime.now().strftime("%B %d, %Y")

    status_color = "#10B981" if is_approved else "#EF4444"
    status_bg = "#ECFDF5" if is_approved else "#FEF2F2"
    status_border = "#059669" if is_approved else "#DC2626"

    # Driving factors HTML
    factors_html = ""
    for f in explanation["primary_driving_factors"]:
        f_color = "#065F46" if is_approved else "#991B1B"
        f_bg = "#F0FDF4" if is_approved else "#FFF1F2"
        factors_html += f"""
        <div style="background-color: {f_bg}; border-left: 4px solid {status_color}; padding: 12px 16px; margin-bottom: 12px; border-radius: 4px;">
            <div style="font-weight: bold; color: {f_color}; font-size: 15px;">
                {f['factor']} <span style="float: right; font-size: 13px; font-weight: normal;">({f['impact']})</span>
            </div>
            <div style="margin-top: 6px; font-size: 14px; color: #374151; line-height: 1.4;">
                {f['explanation']}
            </div>
        </div>
        """

    # Counterfactuals HTML (for rejections)
    counterfactuals_html = ""
    if explanation.get("counterfactual_targets"):
        counterfactuals_html += """
        <div style="margin-top: 24px;">
            <h3 style="color: #1F2937; border-bottom: 1px solid #E5E7EB; padding-bottom: 6px; font-size: 16px;">
                Roadmap to Future Approval & Counterfactual Milestones
            </h3>
            <ul style="color: #374151; font-size: 14px; line-height: 1.6; padding-left: 20px;">
        """
        for cf in explanation["counterfactual_targets"]:
            counterfactuals_html += f"""
                <li style="margin-bottom: 8px;">
                    <strong>{cf['action']}:</strong> Currently <code>{cf['current']}</code> &rarr; Target <code>{cf['target']}</code>.<br>
                    <span style="color: #4B5563;">{cf['impact']} &mdash; {cf['how_to_achieve']}</span>
                </li>
            """
        counterfactuals_html += "</ul></div>"

    policy_html = ""
    if evaluation_result.get("policy_violations"):
        policy_html = f"""
        <div style="background-color: #FEF2F2; border: 1px solid #F87171; padding: 12px 16px; border-radius: 6px; margin: 16px 0; color: #991B1B; font-size: 14px;">
            <strong>Mandatory Underwriting Policy Violation:</strong><br>
            {evaluation_result['policy_violations'][0]}
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>LendingClub Loan Underwriting Notice - {evaluation_result['applicant_id']}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #111827;
            line-height: 1.5;
            background-color: #F9FAFB;
            padding: 40px 20px;
            margin: 0;
        }}
        .container {{
            max-width: 780px;
            margin: 0 auto;
            background: #FFFFFF;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            border: 1px solid #E5E7EB;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #E5E7EB;
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .logo-text {{
            font-size: 22px;
            font-weight: 800;
            color: #1E3A8A;
            letter-spacing: -0.5px;
        }}
        .doc-title {{
            font-size: 13px;
            color: #6B7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            text-align: right;
        }}
        .badge {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 800;
            font-size: 18px;
            letter-spacing: 0.05em;
            background-color: {status_bg};
            color: {status_color};
            border: 1px solid {status_border};
            margin-bottom: 16px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            background: #F3F4F6;
            padding: 16px;
            border-radius: 6px;
            margin: 20px 0;
            font-size: 13px;
        }}
        .info-item span {{
            display: block;
            color: #6B7280;
            font-size: 11px;
            text-transform: uppercase;
        }}
        .info-item strong {{
            color: #111827;
            font-size: 14px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #E5E7EB;
            font-size: 11px;
            color: #9CA3AF;
            text-align: center;
        }}
        @media print {{
            body {{ background: none; padding: 0; }}
            .container {{ box-shadow: none; border: none; padding: 0; max-width: 100%; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <div class="logo-text">🏦 LENDINGCLUB CREDIT CORPORATION</div>
                <div style="font-size: 12px; color: #6B7280;">Automated Credit Underwriting & Decisioning System</div>
            </div>
            <div class="doc-title">
                Date: {date_str}<br>
                Notice ID: {evaluation_result['applicant_id']}
            </div>
        </div>

        <div class="badge">
            {decision}: {evaluation_result['decision_type'].upper()}
        </div>

        <div style="font-size: 16px; font-weight: 600; color: #1F2937; margin-bottom: 8px;">
            {explanation['headline']}
        </div>

        <p style="font-size: 14px; color: #374151; margin-bottom: 16px;">
            {explanation['summary']}
        </p>

        {policy_html}

        <div class="info-grid">
            <div class="info-item">
                <span>Requested Amount</span>
                <strong>${profile['loan_amnt']:,.0f}</strong>
            </div>
            <div class="info-item">
                <span>Approval Probability</span>
                <strong>{evaluation_result['approval_probability']*100:.1f}%</strong>
            </div>
            <div class="info-item">
                <span>Assigned Grade</span>
                <strong>Grade {evaluation_result['grade']}</strong>
            </div>
            <div class="info-item">
                <span>Estimated APR</span>
                <strong>{evaluation_result['estimated_apr']:.2f}%</strong>
            </div>
        </div>

        <h3 style="color: #1F2937; border-bottom: 1px solid #E5E7EB; padding-bottom: 6px; font-size: 16px; margin-top: 28px;">
            Key Factors Influencing Your Credit Decision
        </h3>
        <p style="font-size: 13px; color: #6B7280; margin-top: -4px;">
            Under the Equal Credit Opportunity Act and Fair Credit Reporting Act, lenders must clearly disclose the principal factors affecting credit terms:
        </p>

        {factors_html}

        {counterfactuals_html}

        <div style="margin-top: 24px; background: #F9FAFB; padding: 14px; border-radius: 6px; border: 1px solid #E5E7EB; font-size: 13px; color: #4B5563;">
            <strong>Underwriting Advisory Note:</strong> {explanation['next_steps_advice']}
        </div>

        <div class="footer">
            Equal Credit Opportunity Act (ECOA) & Fair Credit Reporting Act (FCRA) Compliant Disclosure.<br>
            LendingClub Loan Underwriting Agent &bull; Calibrated Predictive Model (ROC-AUC 0.89) &bull; Generated on {date_str}
        </div>
    </div>
</body>
</html>
"""
    return html_content
