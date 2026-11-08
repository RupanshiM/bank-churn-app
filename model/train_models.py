import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET = "Exited"
ID_COLS = ["RowNumber", "CustomerId", "Surname"]
CAT_COLS = ["Geography", "Gender"]
NUM_COLS = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary"]
SEED = 42


def split_data(bank):
    bank = bank.drop(columns=ID_COLS, errors="ignore")
    X, y = bank.drop(columns=[TARGET]), bank[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def churn_pipe(clf):
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS)])
    return Pipeline([("prep", prep), ("clf", clf)])


def build_models():
    return {
        "Logistic Regression": churn_pipe(LogisticRegression(max_iter=2000)),
        "Decision Tree": churn_pipe(DecisionTreeClassifier(max_depth=6, random_state=SEED)),
        "kNN": churn_pipe(KNeighborsClassifier(n_neighbors=11)),
        "Naive Bayes": churn_pipe(GaussianNB()),
        "SVM": churn_pipe(SVC(kernel="rbf", probability=True, random_state=SEED)),
    }


def score_model(est, Xs, ys):
    pred, proba = est.predict(Xs), est.predict_proba(Xs)[:, 1]
    return pred, {"Accuracy": accuracy_score(ys, pred), "AUC": roc_auc_score(ys, proba),
                  "Precision": precision_score(ys, pred, zero_division=0),
                  "Recall": recall_score(ys, pred), "F1": f1_score(ys, pred),
                  "MCC": matthews_corrcoef(ys, pred)}


def train_all(bank):
    X_tr, X_te, y_tr, y_te = split_data(bank)
    models = build_models()
    for est in models.values():
        est.fit(X_tr, y_tr)
    return models, X_te, y_te


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bank = pd.read_csv(os.path.join(root, "Churn_Modelling.csv"))
    models, X_te, y_te = train_all(bank)
    pd.concat([X_te, y_te], axis=1).to_csv(os.path.join(root, "test_data.csv"), index=False)
    rows = [{"ML Model Name": n, **score_model(m, X_te, y_te)[1]} for n, m in models.items()]
    print(pd.DataFrame(rows).round(4).to_string(index=False))
