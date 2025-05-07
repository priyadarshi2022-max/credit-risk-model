# Credit Risk Modelling with Machine Learning

This project implements machine learning models to assess **credit risk** using real-world structured financial and customer data. The goal is to predict the loan approval category (`p1`, `p2`, `p3`, `p4`) based on a set of demographic, behavioral, and financial features.

## 📌 Project Objectives

- Develop a robust machine learning pipeline to model **loan approval risk**.
- Use **feature selection**, **multicollinearity reduction**, and **categorical encoding**.
- Evaluate multiple ML models including **Random Forest**, **XGBoost**, and **Decision Trees**.
- Perform **hyperparameter tuning** to maximize model accuracy and generalization.

## 🛠️ Technologies Used

- Python, Pandas, NumPy, Scikit-learn
- XGBoost, Hyperopt
- Statsmodels (for VIF), SciPy (for Chi-square, ANOVA)
- Matplotlib (for visualizations)

## 📊 Model Performance

| Model              | Accuracy (Test) | Best Class F1 Score | Notes                          |
|-------------------|----------------|---------------------|-------------------------------|
| Random Forest      | 0.762          | P2: 0.865           | Strong baseline                |
| XGBoost (tuned)    | 0.779          | P2: 0.865           | Best overall performance       |
| Decision Tree      | 0.697          | P2: 0.801           | Lower generalization           |

> Final tuned XGBoost outperformed all models with ~77.9% accuracy.

## 🔍 Feature Engineering & Selection

- **Missing Value Treatment**: Removed or filtered based on thresholds
- **Categorical Analysis**: Chi-square tests for independence
- **Multicollinearity Check**: Variance Inflation Factor (VIF)
- **Statistical Significance**: ANOVA across target classes

## 📂 Folder Structure
```
credit-risk-modelling/
│
├── data/
│ ├── train.csv
│ ├── val.csv
│ ├── test.csv
│
├── notebooks/
│ ├── 1_data_preprocessing.ipynb
│ ├── 2_model_training.ipynb
│ └── 3_hyperparameter_tuning.ipynb
│
├── hyperparameters.csv
├── README.md
└── requirements.txt
```
