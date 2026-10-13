import pandas as pd
import os

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_PATH = "data/processed_data.csv"
OUTPUT_PATH = "generated_data/synthetic_copula.csv"

NUM_ROWS = 569


print("==========================================")
print("   GAUSSIAN COPULA SYNTHETIC DATA")
print("==========================================")


# ==========================================
# 1. Check input dataset
# ==========================================

if not os.path.exists(INPUT_PATH):
    print("\nERROR: Processed dataset not found!")
    print("Expected:", INPUT_PATH)
    exit()

df = pd.read_csv(INPUT_PATH)

print("\nOriginal dataset shape:")
print(df.shape)


# ==========================================
# 2. Create SDV metadata
# ==========================================

print("\nCreating dataset metadata...")

metadata = SingleTableMetadata()

metadata.detect_from_dataframe(data=df)

# Explicitly set diagnosis as categorical
metadata.update_column(
    column_name="diagnosis",
    sdtype="categorical"
)

print("Metadata created successfully.")


# ==========================================
# 3. Display detected metadata
# ==========================================

print("\nDetected column types:")

for column, details in metadata.to_dict()["columns"].items():
    print(f"{column}: {details['sdtype']}")


# ==========================================
# 4. Create Gaussian Copula model
# ==========================================

print("\nCreating Gaussian Copula synthesizer...")

synthesizer = GaussianCopulaSynthesizer(
    metadata
)

print("Synthesizer created successfully.")


# ==========================================
# 5. Train synthesizer
# ==========================================

print("\nTraining Gaussian Copula model...")
print("Please wait...")

synthesizer.fit(df)

print("Training completed successfully.")


# ==========================================
# 6. Generate synthetic data
# ==========================================

print("\nGenerating synthetic data...")

synthetic_data = synthesizer.sample(
    num_rows=NUM_ROWS
)

print("Synthetic data generated successfully.")


# ==========================================
# 7. Create output directory
# ==========================================

os.makedirs("generated_data", exist_ok=True)


# ==========================================
# 8. Save synthetic dataset
# ==========================================

synthetic_data.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSynthetic dataset saved to:")
print(OUTPUT_PATH)


# ==========================================
# 9. Display result
# ==========================================

print("\n==========================================")
print("       SYNTHETIC DATA INFORMATION")
print("==========================================")

print("\nRows:", synthetic_data.shape[0])
print("Columns:", synthetic_data.shape[1])

print("\nFirst 5 synthetic records:")
print(synthetic_data.head())

print("\nDiagnosis distribution:")
print(synthetic_data["diagnosis"].value_counts())

print("\n==========================================")
print("   GAUSSIAN COPULA COMPLETED")
print("==========================================")