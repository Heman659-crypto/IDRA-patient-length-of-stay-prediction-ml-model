# IDRA Healthcare Length of Stay Prediction

> **IDRA Data Science & AI Capstone Project 2026**

## Predictive Analytics for Healthcare: Forecasting Patient Length of Stay in Urban Hospitals

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E)](https://scikit-learn.org/)
[![Project](https://img.shields.io/badge/Project-IDRA%20Capstone-6f42c1)](https://github.com/)

## Overview

This repository contains an end-to-end **supervised machine learning regression project** for predicting **patient length of stay (LOS)** in a hospital setting.

The project investigates whether demographic, clinical, prior-utilisation, temporal, and facility-related information available at or near admission can be used to estimate a patient's expected hospital stay.

The analytical workflow is designed as a reproducible capstone record:

**Problem Definition → Data Validation → Data Cleaning → Leakage Assessment → Exploratory Data Analysis → Feature Engineering → Model Development → Evaluation → Interpretation → Conclusions**

> **Important:** This project is an academic analytical proof of concept. Model predictions should not be treated as clinical decisions or as a validated clinical decision-support system.

---

## Project Objective

The primary objective is to develop and evaluate regression models for the target variable:

```text
lengthofstay
```

The project aims to:

1. Understand the structure and quality of the supplied healthcare dataset.
2. Identify and treat demonstrable data-quality issues.
3. Investigate potential target leakage before modelling.
4. Explore relationships between patient characteristics and LOS.
5. Establish a simple baseline and train multiple regression models.
6. Compare model performance using appropriate regression metrics.
7. Interpret important predictors without making causal claims.
8. Document limitations and considerations relevant to healthcare use.

---

## Research Questions

- What does the distribution of patient length of stay look like?
- Which variables show the strongest descriptive relationships with LOS?
- Does a machine learning model improve meaningfully over a simple baseline?
- How do linear and non-linear regression approaches compare?
- Which predictors contribute most strongly to the Random Forest model?
- What limitations should be considered before applying the findings to a real healthcare environment?

---

## Dataset

### Expected dataset

```text
P_1_LengthOfStay.csv
```

The notebook validates the file and required schema before analysis.

### Core variables

| Variable | Role / interpretation | Modelling treatment |
|---|---|---|
| `eid` | Encounter/row identifier | Excluded |
| `lengthofstay` | Target: hospital stay in days | Prediction target |
| `vdate` | Admission / visit date | Used to derive admission-time features |
| `discharged` | Discharge date | Excluded because it can reveal future information / encode LOS |
| `rcount` | Prior admission/readmission count | Converted to numeric/ordinal representation |
| `gender` | Patient category | One-hot encoded |
| `facid` | Facility identifier/category | One-hot encoded |
| Clinical variables | Patient clinical measurements | Used according to data availability and quality |

### Dataset handling policy

The repository is configured **not to commit the raw CSV by default**.

This is intentional because course/LMS-provided datasets may have redistribution restrictions. Place the dataset locally at:

```text
data/P_1_LengthOfStay.csv
```

or update the notebook configuration to match your local setup.

**Do not upload academic/private datasets to GitHub unless the dataset owner explicitly permits redistribution.**

---

## Machine Learning Approach

### Problem type

**Supervised Machine Learning — Regression**

### Target

```text
lengthofstay
```

### Models

#### 1. Dummy Mean Baseline

Provides a reference point by predicting the training-set mean LOS.

#### 2. Linear Regression

Provides an interpretable linear modelling baseline.

#### 3. Random Forest Regressor

Captures non-linear relationships and interactions between predictors.

### Evaluation metrics

The project uses:

- **MAE — Mean Absolute Error**
- **RMSE — Root Mean Squared Error**
- **R² — Coefficient of Determination**

Five-fold cross-validation is also used on the training set to assess model stability.

---

## Leakage Prevention

A central part of this project is distinguishing variables that would realistically be available at prediction time from variables that contain future information.

The `discharged` variable is therefore investigated for leakage because the discharge date can be directly related to the target LOS:

```text
length of stay ≈ discharge date − admission date
```

The modelling pipeline excludes future-derived information.

Preprocessing steps that learn values from the data, including imputation and categorical encoding, are performed inside scikit-learn pipelines after the train/test split.

This reduces the risk of information from the test data influencing model training.

---

## Project Structure

```text
idra-patient-length-of-stay-prediction/
│
├── notebooks/
│   └── IDRA_LengthOfStay_Capstone_Professional.ipynb
│
├── data/
│   └── README.md
│   └── P_1_LengthOfStay.csv        # local only; normally ignored by Git
│
├── reports/
│   └── README.md
│
├── src/
│   └── README.md
│
├── .github/
│   └── ISSUE_TEMPLATE/
│       └── bug_report.md
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Notebook Workflow

The main notebook follows this sequence:

1. Project Overview
2. Problem Statement
3. Objectives and Research Questions
4. Dataset and Data Dictionary
5. Environment and Reproducibility
6. Data Loading and Validation
7. Data Quality Assessment
8. Data Cleaning and Preparation
9. Target Leakage Investigation
10. Feature Engineering
11. Exploratory Data Analysis
12. Statistical Analysis
13. Machine Learning Methodology
14. Model Development
15. Model Evaluation
16. Model Interpretation
17. Results and Findings
18. Limitations and Ethical Considerations
19. Recommendations and Next Steps
20. Conclusion
21. Reproducibility Checklist

---

## Reproducibility

### Recommended environment

- Python 3.10 or newer
- Jupyter Notebook or Google Colab
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install dependencies with:

```bash
pip install -r requirements.txt
```

Then open:

```text
notebooks/IDRA_LengthOfStay_Capstone_Professional.ipynb
```

### Google Colab

For Colab:

1. Upload/open the notebook.
2. Upload `P_1_LengthOfStay.csv`.
3. Run the notebook from top to bottom.
4. Review generated tables, figures, metrics, and findings.

Do not rely on hidden notebook state or previously executed cells.

---

## Data Quality Checks

The notebook checks, where applicable:

- Dataset existence
- Required columns
- Data types
- Missing values
- Duplicate rows
- Invalid target values
- Invalid measurement values
- Categorical value consistency
- Numeric outliers
- Date parsing
- Target consistency
- Leakage risk

Outliers are investigated as potential data-quality or clinical signals; they are not automatically removed simply because they fall outside an IQR threshold.

---

## Feature Engineering

The notebook derives admission-time temporal information from `vdate`, including:

- Admission month
- Admission day of week

The original date string is not used directly as a raw categorical predictor.

The `discharged` date is excluded from modelling because it represents information unavailable at admission and may encode the target.

---

## Exploratory Analysis

The notebook examines:

- LOS distribution
- LOS summary statistics
- LOS by prior admission count
- LOS by facility
- Numeric correlations
- Selected clinical variables
- Comorbidity indicators where present
- Distributional and outlier patterns

Visualizations are designed to answer analytical questions rather than provide decoration.

---

## Model Interpretation

For the Random Forest model, feature importance is extracted from the fitted preprocessing-and-model pipeline.

These values are interpreted as **model-based importance measures**, not causal effects.

A variable being important to a predictive model does not establish that changing that variable would cause LOS to change.

---

## Results

The notebook computes the actual model results dynamically from the supplied dataset.

The repository intentionally does **not** hard-code performance numbers into this README, because the final metrics should correspond to the exact dataset and reproducible execution used for the submitted project.

After running the notebook, the final results section should be used as the authoritative project result.

---

## Limitations

Important limitations include:

- Dataset provenance and representativeness should be verified against the original IDRA/LMS documentation.
- A single supplied dataset may not represent all hospitals or patient populations.
- Observational relationships do not establish causality.
- Random train/test splitting may not reflect temporal deployment conditions.
- Facility identifiers may capture institution-specific patterns that may not generalise to unseen facilities.
- Model performance alone does not establish clinical usefulness.
- Real deployment would require external validation, monitoring, governance, and domain-expert review.

---

## Ethical Considerations

Healthcare prediction systems can affect patients and operational decisions. Any practical use would require careful attention to:

- Privacy and data governance
- Bias and fairness
- External validation
- Explainability
- Clinical oversight
- Appropriate use of predictions
- Monitoring for performance drift

The models in this project are presented for **academic analysis**, not autonomous medical decision-making.

---

## Future Work

Potential extensions include:

- Temporal validation using later admissions as a holdout period.
- Hyperparameter tuning with nested or carefully separated validation.
- Gradient boosting models.
- Prediction intervals or probabilistic forecasting.
- External validation on another hospital dataset.
- SHAP-based model explanation.
- Calibration and subgroup performance analysis.
- Cost-sensitive operational evaluation.
- Deployment as a monitored API or dashboard only after appropriate validation and governance.

---

## Repository Scope

This repository focuses on the **data science and machine learning workflow** of the IDRA capstone.

It is intentionally kept separate from any private LMS materials, credentials, or non-redistributable source data.

---

## Author

**Heman Dhupper**

B.Tech — Computer Science & Engineering  
GNDU Regional Campus Gurdaspur  
IDRA Data Science & AI Capstone Project 2026

---

## Academic Note

This repository is intended for educational and portfolio documentation. All dataset use, redistribution, and publication should comply with the terms and academic rules associated with the original dataset and IDRA programme.
