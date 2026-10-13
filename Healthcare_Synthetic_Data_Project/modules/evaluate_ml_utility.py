import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ==========================================
# FILE PATHS
# ==========================================

REAL_PATH = "data/processed_data.csv"
COPULA_PATH = "generated_data/synthetic_copula.csv"
CTGAN_PATH = "generated_data/synthetic_ctgan.csv"

RESULT_PATH = "results/ml_utility_results.csv"


# ==========================================
# LOAD DATA
# ==========================================

real = pd.read_csv(REAL_PATH)
copula = pd.read_csv(COPULA_PATH)
ctgan = pd.read_csv(CTGAN_PATH)


TARGET = "diagnosis"


# ==========================================
# PREPARE REAL DATA
# ==========================================

X_real = real.drop(columns=[TARGET])
y_real = real[TARGET]

X_copula = copula.drop(columns=[TARGET])
y_copula = copula[TARGET]

X_ctgan = ctgan.drop(columns=[TARGET])
y_ctgan = ctgan[TARGET]


# ==========================================
# SPLIT REAL DATA
# ==========================================

X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
    X_real,
    y_real,
    test_size=0.20,
    random_state=42,
    stratify=y_real
)


# ==========================================
# MODEL
# ==========================================

def create_model():

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    return model


# ==========================================
# EVALUATION FUNCTION
# ==========================================

def evaluate_model(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test
):

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # Identify positive class
    positive_class = "M"

    # RandomForest classes are alphabetically ordered:
    # B, M
    positive_index = list(
        model.named_steps[
            "classifier"
        ].classes_
    ).index(positive_class)

    probabilities = model.predict_proba(
        X_test
    )[:, positive_index]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label=positive_class
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=positive_class
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label=positive_class
    )

    y_test_binary = (
        y_test == positive_class
    ).astype(int)

    auc = roc_auc_score(
        y_test_binary,
        probabilities
    )

    return {
        "Experiment": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": auc
    }


# ==========================================
# RUN EXPERIMENTS
# ==========================================

print("==========================================")
print("           ML UTILITY EVALUATION")
print("==========================================")

print("\nReal dataset:")
print(real.shape)

print("\nCopula dataset:")
print(copula.shape)

print("\nCTGAN dataset:")
print(ctgan.shape)

print("\nReal test set:")
print(X_test_real.shape)


# ==========================================
# 1. REAL -> REAL BASELINE
# ==========================================

print("\nTraining Real -> Real model...")

real_result = evaluate_model(
    "Real -> Real",
    X_train_real,
    y_train_real,
    X_test_real,
    y_test_real
)


# ==========================================
# 2. COPULA -> REAL
# ==========================================

print("Training Copula -> Real model...")

copula_result = evaluate_model(
    "Gaussian Copula -> Real",
    X_copula,
    y_copula,
    X_test_real,
    y_test_real
)


# ==========================================
# 3. CTGAN -> REAL
# ==========================================

print("Training CTGAN -> Real model...")

ctgan_result = evaluate_model(
    "CTGAN -> Real",
    X_ctgan,
    y_ctgan,
    X_test_real,
    y_test_real
)


# ==========================================
# CREATE RESULTS TABLE
# ==========================================

results = pd.DataFrame([
    real_result,
    copula_result,
    ctgan_result
])


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("              ML RESULTS")
print("==========================================")

print(
    results.to_string(
        index=False
    )
)


print("\n==========================================")
print("           ML SCORES (%)")
print("==========================================")

for _, row in results.iterrows():

    print(
        f"\n{row['Experiment']}"
    )

    print(
        f"Accuracy : {row['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision: {row['Precision'] * 100:.2f}%"
    )

    print(
        f"Recall   : {row['Recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score : {row['F1_Score'] * 100:.2f}%"
    )

    print(
        f"ROC-AUC  : {row['ROC_AUC'] * 100:.2f}%"
    )


# ==========================================
# SAVE RESULTS
# ==========================================

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

print("\n==========================================")
print("       ML UTILITY COMPLETED")
print("==========================================")