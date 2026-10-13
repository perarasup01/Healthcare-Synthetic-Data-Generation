import pandas as pd
import numpy as np
import os
REAL_PATH = "data/processed_data.csv"
COPULA_PATH = "generated_data/synthetic_copula.csv"
CTGAN_PATH = "generated_data/synthetic_ctgan.csv"
RESULT_PATH = "results/validity_results.csv"
real = pd.read_csv(REAL_PATH)
copula = pd.read_csv(COPULA_PATH)
ctgan = pd.read_csv(CTGAN_PATH)
numeric_columns = real.select_dtypes(
    include=np.number
).columns.tolist()
print("\nNumerical features:", len(numeric_columns))
def evaluate_validity(
    synthetic,
    method_name
):
    print(
        f"\nEvaluating {method_name}..."
    )
    total_checks = 0
    passed_checks = 0
    invalid_details = []
    missing_count = (
        synthetic.isnull()
        .sum()
        .sum()
    )
    total_checks += 1
    if missing_count == 0:
        passed_checks += 1
    else:
        invalid_details.append(
            f"Missing values: {missing_count}"
        )
    valid_diagnosis = synthetic[
        "diagnosis"
    ].isin(["B", "M"]).all()

    total_checks += 1
    if valid_diagnosis:
        passed_checks += 1
    else:
        invalid_details.append(
            "Invalid diagnosis values found"
        )
    duplicate_count = (
        synthetic.duplicated()
        .sum()
    )
    total_checks += 1
    if duplicate_count == 0:
        passed_checks += 1
    else:
        invalid_details.append(
            f"Duplicate records: {duplicate_count}"
        )
    range_passes = 0
    range_total = 0
    for column in numeric_columns:
        real_min = real[column].min()
        real_max = real[column].max()
        synthetic_min = synthetic[column].min()
        synthetic_max = synthetic[column].max()
        range_total += 1
        if (
            synthetic_min >= real_min
            and
            synthetic_max <= real_max
        ):
            range_passes += 1
        else:
            invalid_details.append(
                f"{column}: "
                f"synthetic range "
                f"[{synthetic_min:.4f}, "
                f"{synthetic_max:.4f}] "
                f"outside real range "
                f"[{real_min:.4f}, "
                f"{real_max:.4f}]"
            )
    negative_values = 0
    for column in numeric_columns:
        negative_values += (
            synthetic[column] < 0
        ).sum()
    total_checks += 1
    if negative_values == 0:
        passed_checks += 1
    else:
        invalid_details.append(
            f"Negative numerical values: "
            f"{negative_values}"
        )
    range_score = (
        range_passes /
        range_total
        if range_total > 0
        else 0
    )
    basic_score = (
        passed_checks /
        total_checks
    )
    validity_score = (
        range_score * 0.70
        +
        basic_score * 0.30
    )
    print(
        "\nRange-valid features:",
        f"{range_passes}/{range_total}"
    )
    print(
        "Range validity:",
        f"{range_score * 100:.2f}%"
    )
    print(
        "Basic validity:",
        f"{basic_score * 100:.2f}%"
    )
    print(
        "Overall validity:",
        f"{validity_score * 100:.2f}%"
    )
    if invalid_details:

        print("\nPotential validity issues:")

        for issue in invalid_details:
            print("-", issue)
    else:
        print(
            "\nNo validity issues detected."
        )
    return {
        "Method": method_name,
        "Range_Valid_Features":
            range_passes,
        "Total_Numerical_Features":
            range_total,
        "Range_Validity":
            range_score,
        "Basic_Validity":
            basic_score,
        "Validity_Score":
            validity_score
    }
copula_result = evaluate_validity(
    copula,
    "Gaussian Copula"
)
ctgan_result = evaluate_validity(
    ctgan,
    "CTGAN"
)
results = pd.DataFrame([
    copula_result,
    ctgan_result
])
print(
    results.to_string(
        index=False
    )
)
os.makedirs(
    "results",
    exist_ok=True
)
results.to_csv(
    RESULT_PATH,
    index=False
)
print("\nResults saved to:")
print(RESULT_PATH)
