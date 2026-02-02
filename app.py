import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor


# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(
    page_title="ML Trainer + Explainability (Optional SHAP)",
    layout="wide",
)

st.title("📦 ML Trainer App (with optional explainability)")
st.caption("Upload a dataset, train a model, evaluate it, and optionally explain predictions.")


# ---------------------------
# Helpers
# ---------------------------
def is_classification_target(y: pd.Series) -> bool:
    """Heuristic: if few unique values or boolean/object, treat as classification."""
    if y.dtype == "bool":
        return True
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        return True
    # If integer with small unique count, likely class labels
    nunique = y.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(y) and nunique <= 20:
        return True
    # If any dtype but small unique count, could be classification
    if nunique <= 10:
        return True
    return False


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_model(task: str, model_name: str):
    if task == "classification":
        if model_name == "Logistic Regression":
            return LogisticRegression(max_iter=2000)
        if model_name == "Random Forest":
            return RandomForestClassifier(n_estimators=300, random_state=42)
        if model_name == "Gradient Boosting":
            return GradientBoostingClassifier(random_state=42)
        raise ValueError("Unknown model for classification")

    if task == "regression":
        if model_name == "Linear Regression":
            return LinearRegression()
        if model_name == "Random Forest":
            return RandomForestRegressor(n_estimators=300, random_state=42)
        if model_name == "Gradient Boosting":
            return GradientBoostingRegressor(random_state=42)
        raise ValueError("Unknown model for regression")

    raise ValueError("Unknown task type")


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


@st.cache_resource(show_spinner=False)
def train_pipeline(df: pd.DataFrame, target_col: str, task: str, model_name: str, test_size: float, random_state: int):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    preprocessor = build_preprocessor(X)
    model = get_model(task, model_name)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task == "classification" else None
    )

    pipe.fit(X_train, y_train)
    return pipe, X_train, X_test, y_train, y_test


def evaluate(task: str, pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    preds = pipe.predict(X_test)

    if task == "classification":
        # Try probability metrics if available
        proba = None
        if hasattr(pipe.named_steps["model"], "predict_proba"):
            proba = pipe.predict_proba(X_test)[:, 1] if pipe.predict_proba(X_test).shape[1] == 2 else None

        metrics = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision (weighted)": precision_score(y_test, preds, average="weighted", zero_division=0),
            "Recall (weighted)": recall_score(y_test, preds, average="weighted", zero_division=0),
            "F1 (weighted)": f1_score(y_test, preds, average="weighted", zero_division=0),
        }
        if proba is not None:
            try:
                metrics["ROC AUC"] = roc_auc_score(y_test, proba)
            except Exception:
                pass
        return metrics, preds

    # regression
    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": mean_squared_error(y_test, preds, squared=False),
        "R²": r2_score(y_test, preds),
    }
    return metrics, preds


def safe_shap_explain(pipe: Pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame, task: str):
    """
    Computes SHAP values only when requested.
    Designed to avoid huge compute: samples background + explain rows.
    """
    try:
        import shap
    except Exception as e:
        return None, f"SHAP import failed. If deployment is stuck, try removing `shap` first.\n\nError: {e}"

    # Transform features through preprocessing
    preprocess = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    Xb = preprocess.transform(X_background)
    Xe = preprocess.transform(X_explain)

    # Feature names after preprocessing
    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"f{i}" for i in range(Xb.shape[1])])

    # Choose explainer
    try:
        if "Forest" in model.__class__.__name__ or "GradientBoosting" in model.__class__.__name__:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, Xb)
        shap_values = explainer(Xe)
        return (shap_values, feature_names), None
    except Exception as e:
        return None, f"SHAP computation failed: {e}"


# ---------------------------
# Sidebar: workflow + controls
# ---------------------------
with st.sidebar:
    st.header("🧭 Roadmap")
    st.write(
        """
**Phase 1 (now)**  
- Upload data  
- Pick target  
- Train a baseline model  
- Review quick metrics  

**Phase 2 (next)**  
- Data cleanup + feature engineering  
- Cross-validation  
- Hyperparameter tuning  
- Better model comparison  

**Phase 3 (later)**  
- Explainability (SHAP on samples)  
- Monitoring + drift checks  
- Export pipeline + deployment
        """
    )

    st.divider()
    st.header("⚙️ Training settings")
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)

    st.divider()
    st.header("🧪 SHAP settings")
    enable_shap = st.checkbox("Enable SHAP (optional)", value=False)
    shap_background_n = st.slider("Background sample size", 50, 1000, 200, 50, disabled=not enable_shap)
    shap_explain_n = st.slider("Explain sample size", 5, 200, 50, 5, disabled=not enable_shap)


# ---------------------------
# Main: Upload + train
# ---------------------------
uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if not uploaded:
    st.info("Upload a CSV to begin. Nothing trains until you press the Train button.")
    st.stop()

try:
    df = load_csv(uploaded)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(30), use_container_width=True)

if df.shape[1] < 2:
    st.error("Your dataset needs at least 2 columns (features + target).")
    st.stop()

target_col = st.selectbox("Select target column", options=df.columns.tolist())

y = df[target_col]
task = "classification" if is_classification_target(y) else "regression"
st.write(f"Detected task: **{task}**")

if task == "classification":
    model_name = st.selectbox("Choose model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
else:
    model_name = st.selectbox("Choose model", ["Linear Regression", "Random Forest", "Gradient Boosting"])

train_clicked = st.button("🚀 Train model", type="primary")

if not train_clicked:
    st.caption("Tip: Training and SHAP only run after you click **Train model**.")
    st.stop()

# Basic cleaning: drop rows with missing target
df_train = df.dropna(subset=[target_col]).copy()
if df_train.shape[0] != df.shape[0]:
    st.warning(f"Dropped {df.shape[0] - df_train.shape[0]} rows with missing target.")

with st.spinner("Training pipeline..."):
    pipe, X_train, X_test, y_train, y_test = train_pipeline(
        df_train, target_col, task, model_name, test_size, random_state
    )

metrics, preds = evaluate(task, pipe, X_test, y_test)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📈 Metrics")
    metric_df = pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())})
    st.dataframe(metric_df, use_container_width=True)

with col2:
    st.subheader("🔎 Test set snapshot")
    preview = X_test.copy()
    preview["y_true"] = y_test.values
    preview["y_pred"] = preds
    st.dataframe(preview.head(20), use_container_width=True)

st.divider()

# ---------------------------
# Optional: Predict on new data
# ---------------------------
st.subheader("🧾 Predict on new data (optional)")
st.caption("Upload another CSV with the same feature columns (target column can be absent).")

new_file = st.file_uploader("Upload CSV for prediction", type=["csv"], key="pred_uploader")
if new_file:
    try:
        df_new = pd.read_csv(new_file)
        if target_col in df_new.columns:
            df_new = df_new.drop(columns=[target_col])

        # Align columns (basic check)
        missing_cols = [c for c in X_train.columns if c not in df_new.columns]
        extra_cols = [c for c in df_new.columns if c not in X_train.columns]

        if missing_cols:
            st.error(f"Prediction file is missing columns: {missing_cols}")
        else:
            if extra_cols:
                st.warning(f"Ignoring extra columns not seen in training: {extra_cols}")
                df_new = df_new[X_train.columns]

            with st.spinner("Predicting..."):
                new_preds = pipe.predict(df_new)
            out = df_new.copy()
            out["prediction"] = new_preds

            st.success("Predictions generated.")
            st.dataframe(out.head(50), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download predictions as CSV",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv",
            )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()

# ---------------------------
# Optional: SHAP
# ---------------------------
st.subheader("🧠 Explainability (SHAP, optional)")
if not enable_shap:
    st.info("Enable SHAP from the sidebar if you want explanations. (This can make deployments heavier.)")
    st.stop()

# Sample to keep it fast
bg = X_train.sample(n=min(shap_background_n, len(X_train)), random_state=random_state)
ex = X_test.sample(n=min(shap_explain_n, len(X_test)), random_state=random_state)

with st.spinner("Computing SHAP values on a sample..."):
    result, err = safe_shap_explain(pipe, bg, ex, task)

if err:
    st.error(err)
    st.stop()

(shap_values, feature_names) = result

st.caption("Showing SHAP summary for a small sample (to keep things fast).")

try:
    import shap
    import matplotlib.pyplot as plt

    fig = plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=20)
    st.pyplot(fig, clear_figure=True)

    st.write("Sample rows explained:")
    st.dataframe(ex.reset_index(drop=True), use_container_width=True)

except Exception as e:
    st.error(f"Could not render SHAP plots: {e}")
