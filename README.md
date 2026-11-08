# Bank Customer Churn Prediction

## a. Problem Statement
Banks lose revenue when customers close their accounts. The goal of this project is to predict whether a bank customer will leave the bank (**Exited = 1**) or stay (**Exited = 0**) from their profile and account details, and to compare five classification models on this task using six evaluation metrics. The models are served through an interactive Streamlit app.

## b. Dataset Description
- **Source:** Kaggle – Bank Customer Churn (`Churn_Modelling.csv`)
- **Instances:** 10,000 customers
- **Target:** `Exited` (binary). About 20% of customers churned (2,037) and 80% stayed (7,963), so the classes are imbalanced.
- **Columns:** 13 input columns. `RowNumber`, `CustomerId` and `Surname` are identifiers and were dropped, leaving 10 raw features:

| Feature | Type | Description |
|---|---|---|
| CreditScore | Numeric | Customer's credit score |
| Geography | Categorical | France, Germany or Spain |
| Gender | Categorical | Male / Female |
| Age | Numeric | Customer's age |
| Tenure | Numeric | Years with the bank |
| Balance | Numeric | Account balance |
| NumOfProducts | Numeric | Number of bank products held |
| HasCrCard | Binary | Has a credit card (1/0) |
| IsActiveMember | Binary | Active member (1/0) |
| EstimatedSalary | Numeric | Estimated annual salary |

- **Preprocessing:** Numeric features are standardised with `StandardScaler`. `Geography` (3 values) and `Gender` (2 values) are one-hot encoded, so the models are trained on **13 features**. There are no missing values.
- **Split:** 80/20 stratified train/test split. The 20% split (2,000 rows) is saved as `test_data.csv`.

## c. GitHub Repository Link
https://github.com/RupanshiM/bank-churn-app

**Live Streamlit app:** https://bank-churn-rupanshi.streamlit.app

### Repository structure
```
bank-churn-app/
├── app.py                 # Streamlit web app
├── requirements.txt       # Python dependencies
├── README.md
├── Churn_Modelling.csv    # Full dataset
├── test_data.csv          # 20% held-out test data
└── model/
    └── train_models.py    # Preprocessing, the 5 models and metric calculation
```

## d. Models Used
All five models are built as scikit-learn pipelines (preprocessing + classifier) and trained on the same training split:
1. Logistic Regression
2. Decision Tree Classifier (max_depth = 6)
3. K-Nearest Neighbors (k = 11)
4. Gaussian Naive Bayes
5. Support Vector Machine (RBF kernel)

### Comparison Table (on test_data.csv, 2,000 customers)
Precision, Recall and F1 are for the churn class (Exited = 1).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8080 | 0.7748 | 0.5891 | 0.1867 | 0.2836 | 0.2515 |
| Decision Tree | 0.8605 | **0.8402** | 0.7759 | **0.4423** | **0.5634** | 0.5150 |
| kNN | 0.8420 | 0.8123 | 0.7571 | 0.3292 | 0.4589 | 0.4284 |
| Naive Bayes | 0.8060 | 0.7773 | 0.5331 | 0.3759 | 0.4409 | 0.3351 |
| Support Vector Machine | **0.8625** | 0.8251 | **0.8548** | 0.3907 | 0.5363 | **0.5180** |

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Reaches 80.8% accuracy, but that is barely above the 79.6% you would get by predicting "stayed" for everyone. It catches only 18.7% of churners (recall 0.187) and has the lowest MCC (0.25). A single linear boundary cannot capture the non-linear patterns in churn, such as risk rising sharply with age and with 3–4 products. |
| Decision Tree | The strongest model at actually finding churners: best recall (0.442), best F1 (0.563) and best AUC (0.840), with MCC (0.515) almost equal to SVM. Its splits naturally pick up thresholds and interactions (age, number of products, active membership, Germany). Limiting depth to 6 keeps it from overfitting. |
| kNN | A middle performer (accuracy 0.842, MCC 0.428). Precision is good (0.757), but recall is low (0.329) because churners are the minority, so most of the 11 nearest neighbours of a churner tend to be customers who stayed. Scaling the features was essential for distance-based voting. |
| Naive Bayes | The lowest accuracy (0.806), but it finds twice as many churners as Logistic Regression (recall 0.376) at the cost of more false alarms (precision 0.533). Its independence assumption does not hold well here (e.g. Balance and NumOfProducts are related), which limits its overall MCC (0.335). |
| Support Vector Machine | Highest accuracy (0.8625), highest precision (0.855) and highest MCC (0.518). The RBF kernel models non-linear boundaries well, so when it flags a customer as churning it is usually right. However, it is conservative and still misses about 61% of churners (recall 0.391). |
| **Overall winner** | **Decision Tree** for this dataset. SVM and Decision Tree are effectively tied on MCC (0.518 vs 0.515), but in churn prediction missing a customer who is about to leave is the costlier mistake. The Decision Tree catches the most churners and has the best F1 and AUC. SVM is the better choice if false alarms must be kept to a minimum. Because the data is imbalanced, accuracy alone is misleading: MCC, F1 and recall separate the models far better. |

## Streamlit App Features
- Upload test data as CSV (a download button for `test_data.csv` is included in the sidebar)
- Model selection dropdown for all 5 models
- All six evaluation metrics displayed for the selected model
- Confusion matrix and classification report
- Optional side-by-side comparison of all 5 models with a bar chart

## How to Run Locally
```
pip install -r requirements.txt
python model/train_models.py   # prints the comparison table and writes test_data.csv
streamlit run app.py
```
