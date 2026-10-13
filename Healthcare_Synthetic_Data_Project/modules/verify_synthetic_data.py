import pandas as pd
import os


REAL_PATH = "data/processed_data.csv"
COPULA_PATH = "generated_data/synthetic_copula.csv"
CTGAN_PATH = "generated_data/synthetic_ctgan.csv"


def verify_dataset(name, path):

    print("\n==========================================")
    print(name)
    print("==========================================")

    if not os.path.exists(path):
        print("ERROR: File not found:")
        print(path)
        return

    df = pd.read_csv(path)

    print("\nShape:")
    print(df.shape)

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nDiagnosis distribution:")
    print(df["diagnosis"].value_counts())

    print("\nDiagnosis percentage:")
    print(df["diagnosis"].value_counts(normalize=True) * 100)

    print("\nNumerical columns with negative values:")

    numeric_columns = df.select_dtypes(include="number").columns

    negative_found = False

    for column in numeric_columns:

        if (df[column] < 0).any():

            print(
                column,
                "->",
                (df[column] < 0).sum(),
                "negative values"
            )

            negative_found = True

    if not negative_found:
        print("None")


print("==========================================")
print("     SYNTHETIC DATA VERIFICATION")
print("==========================================")


verify_dataset(
    "REAL DATASET",
    REAL_PATH
)

verify_dataset(
    "GAUSSIAN COPULA DATASET",
    COPULA_PATH
)

verify_dataset(
    "CTGAN DATASET",
    CTGAN_PATH
)


print("\n==========================================")
print("       VERIFICATION COMPLETED")
print("==========================================")