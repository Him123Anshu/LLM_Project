# 📑 Institutional Model Card: LendingClub Loan Underwriting Scorecard

In accordance with Federal Reserve **SR 11-7 Model Risk Management Guidance** and the Mitchell et al. Model Card standard.

---

## 1. Model Details

* **Model Name:** LendingClub Calibrated Credit Underwriting Scorecard (LC-CUS-v1)
* **Model Type:** L2-Regularized Logistic Regression Scorecard with Scaled & One-Hot Transformed Pipeline
* **Framework:** Scikit-Learn 1.8.0 / Python 3.14
* **Decision Threshold:** $\ge 0.55$ Approval Probability
* **Explainability Engine:** Closed-Form Exact Log-Odds Attribution ($\Delta z_i = \beta_i \cdot x_i$)
* **Generative AI Layer:** Google Gemini (`gemini-1.5-flash` / `gemini-2.0-flash`) via `google-generativeai` for regulatory synthesis and conversational underwriting Q&A

---

## 2. Intended Use & Target Audience

* **Intended Use:** Automated, fair, and explainable decisioning for consumer unsecured installment loan applications.
* **Deliverables Generated:**
  1. Binary Eligibility Decision (`APPROVED` vs. `REJECTED`)
  2. LendingClub Credit Risk Grade (`A` through `G`)
  3. Risk-Based Pricing (Estimated APR % and monthly amortization installment)
  4. Regulation B / FCRA Compliant Adverse Action Notice with top principal reasons
  5. Actionable Counterfactual Targets ("What-if" remediation roadmap)
* **Out-of-Scope Use:** Commercial real estate loans, syndicated corporate facilities, or automated decisions without human underwriter override capability.

---

## 3. Training & Evaluation Data

* **Source:** Kaggle LendingClub Loan Dataset (representing peer-to-peer personal loans).
* **Sample Size:** 25,000 stratified loan records (80% Train: 20,000; 20% Test: 5,000).
* **Target Variable:** `is_eligible` (1 = Loan Fully Paid / Good Performance; 0 = Charged Off / Default / Severe Delinquency).
* **Feature Schema:**
  * **Continuous Financial:** `annual_inc`, `loan_amnt`, `dti` (debt-to-income), `fico_score`, `revol_bal`, `revol_util`, `loan_to_income`.
  * **Credit Bureau History:** `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `total_acc`, `pub_rec_bankruptcies`, `emp_length_years`.
  * **Categorical:** `home_ownership` (MORTGAGE, RENT, OWN), `purpose`, `term` (36 months, 60 months).

---

## 4. Performance Metrics (Held-Out Test Set)

| Metric | Score | Industry Benchmark | Compliance Status |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | **`0.8896`** | $> 0.80$ | Exceeds Institutional Tier-1 Benchmark |
| **Precision (Approval)** | **`97.03%`** | $> 90.0\%$ | Extreme Capital Preservation (Low Default Risk) |
| **Recall (Approval)** | **`77.76%`** | $> 70.0\%$ | Balanced Credit Availability |
| **Accuracy** | **`78.70%`** | $> 75.0\%$ | Satisfactory Balanced Classification |
| **Inference Latency** | **`< 1.0 ms`** | $< 100\text{ ms}$ | Real-Time Interactive / Edge Ready |

---

## 5. Fair Lending & Regulatory Compliance

* **Exclusion of Protected Attributes:**
  Under the **Equal Credit Opportunity Act (ECOA, 15 U.S.C. 1691 et seq.)** and **Regulation B (12 CFR Part 1002)**, credit models must not consider race, color, religion, national origin, sex, marital status, or age. The model features strictly evaluate objective financial, credit bureau, and repayment performance variables.
* **Adverse Action Disclosures (FCRA & ECOA):**
  When an applicant is declined, the agent extracts the negative components of the log-odds vector:
  $$\Delta z_i = \beta_i \cdot \left(\frac{v_i - \mu_i}{\sigma_i}\right)$$
  The 4 factors with the largest negative magnitude are ranked and translated into consumer-friendly Adverse Action disclosures.
* **Monotonicity Guarantees:**
  Unlike decision tree ensembles that can produce non-monotonic score drops, this scorecard guarantees that paying down debt, increasing income, or raising credit score will *never* lower a borrower's approval probability.

---

## 6. Model Limitations & Underwriter Guidance

1. **Self-Reported Income:** The model assumes income provided by the applicant is verified or source-verified. Income verification checks should remain in place prior to loan disbursement.
2. **Macroeconomic Shocks:** Scorecard coefficients reflect historical baseline default dynamics. In periods of extreme macroeconomic stress (e.g. sharp recessions), institutional policy gates should be adjusted via the `UnderwritingPolicy` class without altering underlying mathematical weights.
