import pandas as pd
import os

from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


# ==========================================
# CONFIGURATION
# ==========================================

INPUT_PATH = "data/processed_data.csv"
OUTPUT_PATH = "generated_data/synthetic_ctgan.csv"

NUM_ROWS = 569

# Number of training epochs
EPOCHS = 300


print("==========================================")
print("       CTGAN SYNTHETIC DATA")
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
# 2. Create metadata
# ==========================================

print("\nCreating dataset metadata...")

metadata = SingleTableMetadata()

metadata.detect_from_dataframe(data=df)

# Explicitly define diagnosis as categorical
metadata.update_column(
    column_name="diagnosis",
    sdtype="categorical"
)

print("Metadata created successfully.")


# ==========================================
# 3. Display column types
# ==========================================

print("\nDetected column types:")

for column, details in metadata.to_dict()["columns"].items():
    print(f"{column}: {details['sdtype']}")


# ==========================================
# 4. Create CTGAN synthesizer
# ==========================================

print("\nCreating CTGAN synthesizer...")

synthesizer = CTGANSynthesizer(
    metadata,
    epochs=EPOCHS,
    verbose=True
)

print("CTGAN synthesizer created successfully.")


# ==========================================
# 5. Train CTGAN
# ==========================================

print("\nTraining CTGAN...")
print("Epochs:", EPOCHS)
print("This may take several minutes.")
print("------------------------------------------")

synthesizer.fit(df)

print("\nCTGAN training completed successfully.")


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
# 9. Display information
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
print("          CTGAN COMPLETED")
print("==========================================")