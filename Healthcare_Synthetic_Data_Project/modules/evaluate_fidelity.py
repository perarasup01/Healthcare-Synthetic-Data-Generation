import pandas as pd
import numpy as np
import os

from scipy.stats import ks_2samp


REAL_PATH = "data/processed_data.csv"
COPULA_PATH = "generated_data/synthetic_copula.csv"
CTGAN_PATH = "generated_data/synthetic_ctgan.csv"

SUMMARY_PATH = "results/fidelity_summary.csv"
FEATURE_PATH = "results/fidelity_feature_details.csv"


# ==========================================
# LOAD DATA
# ==========================================

real = pd.read_csv(REAL_PATH)
copula = pd.read_csv(COPULA_PATH)
ctgan = pd.read_csv(CTGAN_PATH)

numeric_columns = real.select_dtypes(
    include=np.number
).columns.tolist()


print("==========================================")
print("       STATISTICAL FIDELITY ANALYSIS")
print("==========================================")

print("\nReal dataset:", real.shape)
print("Copula dataset:", copula.shape)
print("CTGAN dataset:", ctgan.shape)
print("Numerical features:", len(numeric_columns))


# ==========================================
# DISTRIBUTION / KS ANALYSIS
# ==========================================

def calculate_ks_results(synthetic_df, method_name):

    results = []

    for column in numeric_columns:

        real_values = real[column].dropna()
        synthetic_values = synthetic_df[column].dropna()

        ks_statistic, p_value = ks_2samp(
            real_values,
            synthetic_values
        )

        similarity = 1 - ks_statistic

        results.append({
            "Method": method_name,
            "Feature": column,
            "KS_Statistic": ks_statistic,
            "P_Value": p_value,
            "Distribution_Similarity": similarity
        })

    return results


copula_ks = calculate_ks_results(
    copula,
    "Gaussian Copula"
)

ctgan_ks = calculate_ks_results(
    ctgan,
    "CTGAN"
)

feature_results = pd.DataFrame(
    copula_ks + ctgan_ks
)


# ==========================================
# OVERALL DISTRIBUTION SCORE
# ==========================================

copula_distribution = feature_results[
    feature_results["Method"] == "Gaussian Copula"
]["Distribution_Similarity"].mean()

ctgan_distribution = feature_results[
    feature_results["Method"] == "CTGAN"
]["Distribution_Similarity"].mean()


# ==========================================
# CORRELATION SIMILARITY
# ==========================================

def calculate_correlation_similarity(
    synthetic_df
):

    real_corr = real[numeric_columns].corr()

    synthetic_corr = synthetic_df[
        numeric_columns
    ].corr()

    difference = np.abs(
        real_corr.values -
        synthetic_corr.values
    )

    mask = ~np.eye(
        difference.shape[0],
        dtype=bool
    )

    mean_difference = difference[mask].mean()

    similarity = 1 - mean_difference

    return np.clip(
        similarity,
        0,
        1
    )


copula_correlation = calculate_correlation_similarity(
    copula
)

ctgan_correlation = calculate_correlation_similarity(
    ctgan
)


# ==========================================
# DIAGNOSIS DISTRIBUTION
# ==========================================

def calculate_diagnosis_similarity(
    synthetic_df
):

    real_distribution = (
        real["diagnosis"]
        .value_counts(normalize=True)
    )

    synthetic_distribution = (
        synthetic_df["diagnosis"]
        .value_counts(normalize=True)
    )

    categories = set(
        real_distribution.index
    ).union(
        synthetic_distribution.index
    )

    differences = []

    for category in categories:

        real_value = real_distribution.get(
            category,
            0
        )

        synthetic_value = synthetic_distribution.get(
            category,
            0
        )

        differences.append(
            abs(
                real_value -
                synthetic_value
            )
        )

    mean_difference = np.mean(
        differences
    )

    similarity = 1 - mean_difference

    return np.clip(
        similarity,
        0,
        1
    )


copula_diagnosis = calculate_diagnosis_similarity(
    copula
)

ctgan_diagnosis = calculate_diagnosis_similarity(
    ctgan
)


# ==========================================
# OVERALL FIDELITY SCORE
# ==========================================

copula_fidelity = (
    copula_distribution * 0.5
    + copula_correlation * 0.3
    + copula_diagnosis * 0.2
)

ctgan_fidelity = (
    ctgan_distribution * 0.5
    + ctgan_correlation * 0.3
    + ctgan_diagnosis * 0.2
)


summary = pd.DataFrame([
    {
        "Method": "Gaussian Copula",
        "Distribution_Similarity":
            copula_distribution,
        "Correlation_Similarity":
            copula_correlation,
        "Diagnosis_Similarity":
            copula_diagnosis,
        "Fidelity_Score":
            copula_fidelity
    },
    {
        "Method": "CTGAN",
        "Distribution_Similarity":
            ctgan_distribution,
        "Correlation_Similarity":
            ctgan_correlation,
        "Diagnosis_Similarity":
            ctgan_diagnosis,
        "Fidelity_Score":
            ctgan_fidelity
    }
])


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("              FIDELITY RESULTS")
print("==========================================")

print(
    summary.to_string(index=False)
)

print("\n==========================================")
print("          FIDELITY SCORES (%)")
print("==========================================")

for _, row in summary.iterrows():

    print(
        f"{row['Method']}: "
        f"{row['Fidelity_Score'] * 100:.2f}%"
    )


# ==========================================
# SAVE RESULTS
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)

summary.to_csv(
    SUMMARY_PATH,
    index=False
)

feature_results.to_csv(
    FEATURE_PATH,
    index=False
)

print("\nSummary saved to:")
print(SUMMARY_PATH)

print("\nFeature-level results saved to:")
print(FEATURE_PATH)

print("\n==========================================")
print("     FIDELITY ANALYSIS COMPLETED")
print("==========================================")