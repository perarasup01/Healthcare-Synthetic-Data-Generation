import pandas as pd
import os

INPUT_PATH = "data/data.csv"
OUTPUT_PATH = "data/processed_data.csv"

print("==========================================")
print("       DATA PREPROCESSING")
print("==========================================")

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

if not os.path.exists(INPUT_PATH):
    print("ERROR: Dataset not found!")
    print("Expected:", INPUT_PATH)
    exit()

df = pd.read_csv(INPUT_PATH)

print("\nOriginal dataset shape:")
print(df.shape)


# --------------------------------------------------
# 2. Remove completely empty columns
# --------------------------------------------------

empty_columns = df.columns[df.isnull().all()].tolist()

if empty_columns:
    print("\nRemoving completely empty columns:")
    print(empty_columns)

    df = df.drop(columns=empty_columns)
else:
    print("\nNo completely empty columns found.")


# --------------------------------------------------
# 3. Remove identifier column
# --------------------------------------------------

if "id" in df.columns:
    print("\nRemoving identifier column: id")
    df = df.drop(columns=["id"])


# --------------------------------------------------
# 4. Clean column names
# --------------------------------------------------

df.columns = df.columns.str.strip()


# --------------------------------------------------
# 5. Remove duplicate rows
# --------------------------------------------------

duplicates = df.duplicated().sum()

if duplicates > 0:
    print("\nRemoving duplicate rows:", duplicates)
    df = df.drop_duplicates()
else:
    print("\nNo duplicate rows found.")


# --------------------------------------------------
# 6. Check missing values
# --------------------------------------------------

missing_values = df.isnull().sum().sum()

print("\nTotal missing values after cleaning:", missing_values)

if missing_values > 0:
    print("WARNING: Missing values still exist.")
else:
    print("Missing-value check: PASS")


# --------------------------------------------------
# 7. Validate diagnosis column
# --------------------------------------------------

if "diagnosis" not in df.columns:
    print("\nERROR: diagnosis column not found!")
    exit()

print("\nDiagnosis values:")
print(df["diagnosis"].value_counts())

valid_diagnosis = set(df["diagnosis"].unique()).issubset({"B", "M"})

if valid_diagnosis:
    print("Diagnosis validation: PASS")
else:
    print("WARNING: Unexpected diagnosis values found.")


# --------------------------------------------------
# 8. Display final dataset information
# --------------------------------------------------

print("\n==========================================")
print("       FINAL DATASET INFORMATION")
print("==========================================")

print("\nRows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumns:")
for column in df.columns:
    print("-", column)


# --------------------------------------------------
# 9. Save processed dataset
# --------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\n==========================================")
print("PREPROCESSING COMPLETED")
print("==========================================")

print("\nProcessed dataset saved to:")
print(OUTPUT_PATH)