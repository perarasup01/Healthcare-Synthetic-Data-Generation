# Healthcare Synthetic Data Generation and Evaluation Framework

## Overview
This project focuses on generating synthetic healthcare data using **Gaussian Copula** and **CTGAN** and evaluating its quality, machine learning utility, validity, and potential privacy risks.

The framework uses the Breast Cancer Wisconsin Diagnostic dataset to compare synthetic data generation techniques and assess their suitability for healthcare research.

## Objectives
- Generate synthetic healthcare data using Gaussian Copula and CTGAN.
- Evaluate statistical fidelity between real and synthetic data.
- Measure machine learning utility using Random Forest.
- Assess potential privacy risks and data validity.
- Calculate a combined trust score for comparison.

## Dataset
**Breast Cancer Wisconsin Diagnostic Dataset**

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)

- 569 records
- 30 numerical features
- 1 diagnosis column

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Gaussian Copula
- CTGAN
- Git and GitHub

## Modules

**Module 1: Data Preprocessing and Synthetic Data Generation**
- Preprocess the original dataset.
- Generate synthetic datasets using Gaussian Copula and CTGAN.

**Module 2: Quality and Privacy Assessment**
- Evaluate fidelity and machine learning utility.
- Assess potential privacy risks and data validity.
- Calculate a combined trust score.

## Key Results

| Metric | Gaussian Copula | CTGAN |
|---|---:|---:|
| Fidelity | 94.07% | 71.62% |
| ML Accuracy | 89.47% | 41.23% |
| Validity | 100% | 100% |
| Trust Score | 96.11% | 78.20% |

*Results are specific to the selected dataset and the evaluation methods implemented in this project. The privacy assessment is a heuristic, not a formal privacy guarantee.*

## Installation

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Healthcare-Synthetic-Data-Project

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Download the dataset from the UCI link above and place it in the input location expected by the project scripts.

## Execution
Run the Python scripts from the project root in the required order:

1. Dataset preprocessing
2. Synthetic data generation
3. Synthetic data verification
4. Fidelity and machine learning utility evaluation
5. Privacy and validity assessment
6. Trust score calculation

Refer to the individual Python scripts for their exact execution commands.

## Future Enhancements
- Real-versus-synthetic data detection
- Medical report generation and authentication
- Streamlit dashboard integration
- Evaluation using additional healthcare datasets
- Enhanced privacy testing

## Conclusion
This project demonstrates a multi-dimensional approach to synthetic healthcare data generation and evaluation. In the current experiments, Gaussian Copula achieved higher fidelity and machine learning utility than CTGAN. Further evaluation is required to establish performance across other datasets and privacy settings.
