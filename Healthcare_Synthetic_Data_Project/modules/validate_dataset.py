import pandas as pd
import os

DATA_PATH = "data/data.csv"

print("==========================================")
print("     HEALTHCARE DATASET VALIDATION")
print("==========================================")

# Check file
if not os.path.exists(DATA_PATH):
    print("\nERROR: Dataset not found!")
    print("Expected location:", DATA_PATH)
    exit()

# Load dataset
df = pd.read_csv(DATA_PATH)

# Basic information
print("\n[1] DATASET SIZE")
print("------------------------------------------")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])

# Column names
print("\n[2] COLUMN NAMES")
print("------------------------------------------")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

# Data types
print("\n[3] DATA TYPES")
print("------------------------------------------")
print(df.dtypes)

# Missing values
print("\n[4] MISSING VALUES")
print("------------------------------------------")
missing = df.isnull().sum()
print(missing)

print("\nTotal missing values:", missing.sum())

# Duplicate rows
print("\n[5] DUPLICATES")
print("------------------------------------------")
print("Duplicate rows:", df.duplicated().sum())

# Diagnosis distribution
if "diagnosis" in df.columns:

    print("\n[6] DIAGNOSIS DISTRIBUTION")
    print("------------------------------------------")
    print(df["diagnosis"].value_counts())

    print("\nDiagnosis percentage:")
    print(df["diagnosis"].value_counts(normalize=True) * 100)

# ID information
if "id" in df.columns:

    print("\n[7] ID COLUMN")
    print("------------------------------------------")
    print("Unique IDs:", df["id"].nunique())
    print("Total rows:", len(df))

    if df["id"].nunique() == len(df):
        print("ID check: PASS - every row has a unique ID")
    else:
        print("ID check: WARNING - duplicate IDs found")

# Numeric columns
numeric_columns = df.select_dtypes(include="number").columns

print("\n[8] NUMERICAL COLUMNS")
print("------------------------------------------")
print("Number of numerical columns:", len(numeric_columns))

for column in numeric_columns:
    print(column)

# Categorical columns
categorical_columns = df.select_dtypes(include="object").columns

print("\n[9] CATEGORICAL COLUMNS")
print("------------------------------------------")
print("Number of categorical columns:", len(categorical_columns))

for column in categorical_columns:
    print(column)

# Basic statistics
print("\n[10] BASIC STATISTICS")
print("------------------------------------------")
print(df.describe())

print("\n==========================================")
print("       VALIDATION COMPLETED")
print("==========================================")