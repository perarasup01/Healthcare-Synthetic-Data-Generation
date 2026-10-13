import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

REAL_PATH = "data/processed_data.csv"
COPULA_PATH = "generated_data/synthetic_copula.csv"
CTGAN_PATH = "generated_data/synthetic_ctgan.csv"

RESULT_PATH = "results/privacy_risk_results.csv"

real = pd.read_csv(REAL_PATH)
copula = pd.read_csv(COPULA_PATH)
ctgan = pd.read_csv(CTGAN_PATH)

feature_columns = real.select_dtypes(
    include=np.number
).columns.tolist()
print("\nNumerical features:", len(feature_columns))

scaler = StandardScaler()

real_scaled = scaler.fit_transform(
    real[feature_columns]
)

copula_scaled = scaler.transform(
    copula[feature_columns]
)

ctgan_scaled = scaler.transform(
    ctgan[feature_columns]
)

print("\nCalculating real-to-real distances...")

real_nn = NearestNeighbors(
    n_neighbors=2,
    metric="euclidean"
)

real_nn.fit(real_scaled)

real_distances, _ = real_nn.kneighbors(
    real_scaled
)

real_nearest_distances = real_distances[:, 1]

def evaluate_privacy(
    synthetic_data,
    method_name
):

    print(
        f"\nEvaluating {method_name}..."
    )

    nn = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean"
    )

    nn.fit(real_scaled)

    synthetic_distances, _ = nn.kneighbors(
        synthetic_data
    )

    synthetic_nearest_distances = (
        synthetic_distances[:, 0]
    )

    median_distance = np.median(
        synthetic_nearest_distances
    )

    mean_distance = np.mean(
        synthetic_nearest_distances
    )

    minimum_distance = np.min(
        synthetic_nearest_distances
    )

    threshold = np.percentile(
        real_nearest_distances,
        5
    )

    high_risk_count = np.sum(
        synthetic_nearest_distances
        <= threshold
    )

    high_risk_percentage = (
        high_risk_count /
        len(synthetic_nearest_distances)
    ) * 100

    privacy_indicator = (
        1 - high_risk_percentage / 100
    )

    return {
        "Method": method_name,
        "Mean_Nearest_Real_Distance":
            mean_distance,
        "Median_Nearest_Real_Distance":
            median_distance,
        "Minimum_Nearest_Real_Distance":
            minimum_distance,
        "Risk_Threshold":
            threshold,
        "Potentially_Close_Records":
            int(high_risk_count),
        "Potential_Risk_Percentage":
            high_risk_percentage,
        "Privacy_Indicator":
            privacy_indicator
    }

copula_result = evaluate_privacy(
    copula_scaled,
    "Gaussian Copula"
)

ctgan_result = evaluate_privacy(
    ctgan_scaled,
    "CTGAN"
)
results = pd.DataFrame([
    copula_result,
    ctgan_result
])

print(
    results.to_string(index=False)
)
for _, row in results.iterrows():
    print(
        f"\n{row['Method']}"
    )
    print(
        f"Potentially close records: "
        f"{row['Potentially_Close_Records']}"
    )
    print(
        f"Potential risk percentage: "
        f"{row['Potential_Risk_Percentage']:.2f}%"
    )
    print(
        f"Privacy indicator: "
        f"{row['Privacy_Indicator'] * 100:.2f}%"
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
