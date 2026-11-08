import os, sys
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, streamlit as st
from sklearn.metrics import confusion_matrix, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "model"))
from train_models import train_all, score_model, TARGET, ID_COLS

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="wide")
CLASS_NAMES = ["Stayed", "Churned"]


@st.cache_resource(show_spinner="Training the 5 models (first load only, ~30 s)...")
def get_models():
    bank = pd.read_csv(os.path.join(HERE, "Churn_Modelling.csv"))
    return train_all(bank)


models, X_hold, y_hold = get_models()

st.title("🏦 Bank Customer Churn Predictor")
st.caption("Upload the test CSV, pick a model and see how well it spots customers who will leave the bank.")

with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload test data (CSV)", type="csv")
    chosen = st.selectbox("Choose a model", list(models.keys()))
    compare_all = st.checkbox("Also compare all 5 models")
    st.divider()
    st.download_button("⬇️ Download test_data.csv",
                       pd.concat([X_hold, y_hold], axis=1).to_csv(index=False),
                       file_name="test_data.csv", mime="text/csv")

if uploaded is None:
    st.info("👈 Upload test_data.csv from the sidebar to begin. "
            "No file yet? Use the download button in the sidebar first.")
    st.stop()

data = pd.read_csv(uploaded).drop(columns=ID_COLS, errors="ignore")
if TARGET not in data.columns:
    st.error(f"The CSV needs an '{TARGET}' column with the true labels (0 = stayed, 1 = churned).")
    st.stop()

X_test, y_test = data.drop(columns=[TARGET]), data[TARGET]
c1, c2, c3 = st.columns(3)
c1.metric("Customers", len(data))
c2.metric("Churned", int(y_test.sum()))
c3.metric("Churn rate", f"{y_test.mean():.1%}")
with st.expander("Preview uploaded data"):
    st.dataframe(data.head(20))

y_pred, scores = score_model(models[chosen], X_test, y_test)
st.subheader(f"Evaluation metrics: {chosen}")
for col, (k, v) in zip(st.columns(6), scores.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred, labels=[0, 1]), annot=True, fmt="d", cmap="Purples",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    rep = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
    st.dataframe(pd.DataFrame(rep).transpose().round(3))

if compare_all:
    st.subheader("All models on this test data")
    table = pd.DataFrame([{"ML Model Name": n, **score_model(m, X_test, y_test)[1]}
                          for n, m in models.items()]).set_index("ML Model Name").round(4)
    st.dataframe(table.style.highlight_max(axis=0, color="#d9d2f3"))
    st.bar_chart(table[["Accuracy", "Recall", "F1", "MCC"]])
