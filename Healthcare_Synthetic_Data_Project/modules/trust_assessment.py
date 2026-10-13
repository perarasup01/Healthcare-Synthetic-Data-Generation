import pandas as pd
import os


# ============================================================
# TRUST ASSESSMENT
# ============================================================

print("=" * 50)
print("           SYNTHETIC DATA TRUST ASSESSMENT")
print("=" * 50)


# ------------------------------------------------------------
# 1. LOAD PREVIOUS RESULTS
# ------------------------------------------------------------

fidelity_file = "results/fidelity_summary.csv"
utility_file = "results/ml_utility_results.csv"
privacy_file = "results/privacy_risk_results.csv"
validity_file = "results/validity_results.csv"


fidelity = pd.read_csv(fidelity_file)
utility = pd.read_csv(utility_file)
privacy = pd.read_csv(privacy_file)
validity = pd.read_csv(validity_file)


# ------------------------------------------------------------
# 2. DISPLAY COLUMN NAMES
# ------------------------------------------------------------

print("\nLoaded result files successfully.")

print("\nFidelity columns:")
print(fidelity.columns.tolist())

print("\nUtility columns:")
print(utility.columns.tolist())

print("\nPrivacy columns:")
print(privacy.columns.tolist())

print("\nValidity columns:")
print(validity.columns.tolist())


# ------------------------------------------------------------
# 3. METHOD NAMES
# ------------------------------------------------------------

methods = ["Gaussian Copula", "CTGAN"]


# ------------------------------------------------------------
# 4. CREATE TRUST ASSESSMENT
# ------------------------------------------------------------

results = []


for method in methods:

    print("\n" + "-" * 50)
    print("Evaluating:", method)
    print("-" * 50)

    # --------------------------------------------------------
    # FIDELITY
    # --------------------------------------------------------

    fidelity_row = fidelity[
        fidelity["Method"] == method
    ]

    if fidelity_row.empty:
        raise ValueError(f"Fidelity result not found for {method}")

    fidelity_score = float(
        fidelity_row["Fidelity_Score"].iloc[0]
    )


    # --------------------------------------------------------
    # VALIDITY
    # --------------------------------------------------------

    validity_row = validity[
        validity["Method"] == method
    ]

    if validity_row.empty:
        raise ValueError(f"Validity result not found for {method}")

    validity_score = float(
        validity_row["Validity_Score"].iloc[0]
    )


    # --------------------------------------------------------
    # PRIVACY
    # --------------------------------------------------------

    privacy_row = privacy[
        privacy["Method"] == method
    ]

    if privacy_row.empty:
        raise ValueError(f"Privacy result not found for {method}")

    privacy_score = float(
        privacy_row["Privacy_Indicator"].iloc[0]
    )


    # --------------------------------------------------------
    # ML UTILITY
    # --------------------------------------------------------
    #
    # We use the average of Accuracy and ROC-AUC
    # for the synthetic-to-real experiment.
    #
    # Real -> Real is the baseline.
    #
    # --------------------------------------------------------

    if method == "Gaussian Copula":
        utility_method = "Gaussian Copula -> Real"
    else:
        utility_method = "CTGAN -> Real"


    utility_row = utility[
        utility["Experiment"] == utility_method
    ]

    if utility_row.empty:
        raise ValueError(
            f"Utility result not found for {utility_method}"
        )

    accuracy = float(
        utility_row["Accuracy"].iloc[0]
    )

    roc_auc = float(
    utility_row["ROC_AUC"].iloc[0]
)

    utility_score = (accuracy + roc_auc) / 2


    # --------------------------------------------------------
    # 5. TRUST SCORE
    # --------------------------------------------------------

    trust_score = (
        fidelity_score * 0.30
        + utility_score * 0.30
        + privacy_score * 0.20
        + validity_score * 0.20
    )


    # --------------------------------------------------------
    # 6. TRUST LEVEL
    # --------------------------------------------------------

    if trust_score >= 0.85:
        trust_level = "High"

    elif trust_score >= 0.70:
        trust_level = "Moderate"

    else:
        trust_level = "Low"


    results.append({
        "Method": method,
        "Fidelity": fidelity_score,
        "ML_Utility": utility_score,
        "Privacy": privacy_score,
        "Validity": validity_score,
        "Trust_Score": trust_score,
        "Trust_Percentage": trust_score * 100,
        "Trust_Level": trust_level
    })


# ------------------------------------------------------------
# 7. CREATE DATAFRAME
# ------------------------------------------------------------

trust_df = pd.DataFrame(results)


# ------------------------------------------------------------
# 8. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("                    TRUST RESULTS")
print("=" * 70)

for _, row in trust_df.iterrows():

    print("\nMethod:", row["Method"])

    print(
        f"Fidelity       : {row['Fidelity'] * 100:.2f}%"
    )

    print(
        f"ML Utility     : {row['ML_Utility'] * 100:.2f}%"
    )

    print(
        f"Privacy        : {row['Privacy'] * 100:.2f}%"
    )

    print(
        f"Validity       : {row['Validity'] * 100:.2f}%"
    )

    print(
        f"Trust Score    : {row['Trust_Percentage']:.2f}%"
    )

    print(
        f"Trust Level    : {row['Trust_Level']}"
    )


# ------------------------------------------------------------
# 9. SAVE RESULTS
# ------------------------------------------------------------

os.makedirs("results", exist_ok=True)

output_file = "results/trust_assessment.csv"

trust_df.to_csv(
    output_file,
    index=False
)


print("\n")
print("=" * 70)
print("Trust assessment saved to:")
print(output_file)
print("=" * 70)

print("\nTRUST ASSESSMENT COMPLETED")