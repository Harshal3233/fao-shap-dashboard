import io
import time
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import shap
import matplotlib.pyplot as plt

try:
    import joblib
except Exception:
    joblib = None


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="FAO SHAP Dashboard",
    layout="wide",
)

st.title(" FAO SHAP Dashboard")
st.caption("Upload data, train a baseline model, and explain predictions with SHAP (fast + Streamlit-friendly).")


# ----------------------------
# Helpers
# ----------------------------
def _is_classification_target(y: pd.Series) -> bool:
    # Heuristic: few unique values => classification
    nunique = y.dropna().nunique()
    if y.dtype == "object" or y.dtype.name == "category":
        return True
    # If numeric but small unique count, likely classification
    return nunique <= 20


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _get_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame):
    # Works with sklearn >= 1.0
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        # Fallback: not perfect, but prevents crash
        return np.array([f"f{i}" for i in range(X.shape[1])])


def _metric_block_regression(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2_score(y_true, y_pred):.4f}")
    c2.metric("MAE", f"{mean_absolute_error(y_true, y_pred):.4f}")
    c3.metric("RMSE", f"{rmse:.4f}")


def _metric_block_classification(y_true, y_pred, y_proba=None):
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.4f}")
    c2.metric("F1 (weighted)", f"{f1_score(y_true, y_pred, average='weighted'):.4f}")

    # ROC AUC only valid for binary (or multi with ovR, but keep it simple)
    auc_text = "N/A"
    if y_proba is not None:
        try:
            # binary case expected: proba for positive class
            if y_proba.ndim == 1:
                auc_text = f"{roc_auc_score(y_true, y_proba):.4f}"
            elif y_proba.ndim == 2 and y_proba.shape[1] == 2:
                auc_text = f"{roc_auc_score(y_true, y_proba[:, 1]):.4f}"
        except Exception:
            pass
    c3.metric("ROC AUC", auc_text)


@st.cache_data(show_spinner=False)
def _load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_resource(show_spinner=False)
def _train_pipeline(df: pd.DataFrame, target_col: str, task: str, test_size: float, random_state: int, n_estimators: int):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # basic cleaning
    X = X.copy()
    y = y.copy()

    preprocessor = _build_preprocessor(X)

    if task == "regression":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task == "classification" else None
    )

    pipe.fit(X_train, y_train)

    return pipe, X_train, X_test, y_train, y_test


def _safe_sample(X: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(X) <= max_rows:
        return X
    return X.sample(max_rows, random_state=random_state)


def _compute_shap_for_tree_model(pipe: Pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame):
    """
    Computes SHAP values for tree-based model inside a sklearn Pipeline.
    Returns: feature_names, shap_values, X_explain_transformed
    """
    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    # transform
    Xb = pre.transform(X_background)
    Xe = pre.transform(X_explain)
    feature_names = _get_feature_names(pre, X_explain)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(Xe)

    return feature_names, shap_values, Xe


def _plot_shap_summary(feature_names, shap_values, X_transformed, max_display=20, plot_type="bar"):
    """
    Handles both regression and classification SHAP shapes.
    """
    fig = plt.figure(figsize=(10, 6))

    # Classification can return list (one array per class)
    if isinstance(shap_values, list):
        # Pick class 1 if binary, else pick the class with largest mean |SHAP|
        if len(shap_values) == 2:
            sv = shap_values[1]
            title_suffix = " (class 1)"
        else:
            means = [np.mean(np.abs(sv)) for sv in shap_values]
            idx = int(np.argmax(means))
            sv = shap_values[idx]
            title_suffix = f" (class {idx})"
    else:
        sv = shap_values
        title_suffix = ""

    shap.summary_plot(
        sv,
        features=X_transformed,
        feature_names=feature_names,
        plot_type=plot_type,
        max_display=max_display,
        show=False
    )
    plt.title(f"SHAP Summary{title_suffix}")
    plt.tight_layout()
    return fig


# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header(" Controls")

    st.markdown("### 1) Load data")
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])

    st.markdown("### 2) Optional: Load a saved model")
    st.caption("If you already trained a pipeline and saved it with joblib, you can load it here.")
    uploaded_model = st.file_uploader("Upload model (.pkl/.joblib)", type=["pkl", "joblib"])

    st.divider()

    st.markdown("### 3) Training / evaluation")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    n_estimators = st.slider("RandomForest trees", 50, 500, 200, 50)

    st.divider()

    st.markdown("### 4) SHAP compute")
    shap_rows = st.slider("Rows to explain (sampling)", 50, 2000, 400, 50)
    shap_bg_rows = st.slider("Background rows (sampling)", 50, 2000, 200, 50)
    max_display = st.slider("Max features to display", 5, 40, 20, 1)

    st.divider()

    st.markdown("### 5) Roadmap text")
    st.caption("This shows up inside the app as a clean 'next steps' section.")
    roadmap_focus = st.selectbox(
        "Focus area",
        ["Data pipeline", "Modeling", "Explainability", "Deployment", "Monitoring"],
        index=0
    )


# ----------------------------
# Main logic
# ----------------------------
if not uploaded_csv and not uploaded_model:
    st.info("Upload a CSV to begin (or upload a saved model pipeline).")
    st.stop()

df = None
pipe = None
X_train = X_test = y_train = y_test = None

# Load CSV if present
if uploaded_csv:
    try:
        df = _load_csv(uploaded_csv.getvalue())
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.subheader(" Data preview")
    st.dataframe(df.head(30), use_container_width=True)

    if df.shape[1] < 2:
        st.error("Your dataset needs at least 2 columns (features + target).")
        st.stop()

    target_col = st.selectbox("Select target column", df.columns.tolist(), index=len(df.columns) - 1)

    y = df[target_col]
    suggested_task = "classification" if _is_classification_target(y) else "regression"
    task = st.radio("Task", ["regression", "classification"], index=0 if suggested_task == "regression" else 1)

    st.write("")

    # Train pipeline
    with st.spinner("Training model..."):
        try:
            pipe, X_train, X_test, y_train, y_test = _train_pipeline(
                df=df,
                target_col=target_col,
                task=task,
                test_size=test_size,
                random_state=random_state,
                n_estimators=n_estimators
            )
        except Exception as e:
            st.error("Training failed. Common causes: non-numeric target for regression, empty columns, or mixed types.")
            st.exception(e)
            st.stop()

    st.success("Model trained ")

    # Evaluate
    st.subheader("📈 Model performance")
    try:
        y_pred = pipe.predict(X_test)

        if task == "regression":
            _metric_block_regression(y_test, y_pred)
        else:
            # Predict probabilities if available
            y_proba = None
            try:
                y_proba = pipe.predict_proba(X_test)
            except Exception:
                pass
            _metric_block_classification(y_test, y_pred, y_proba=y_proba)

    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        st.stop()

    # Offer download of trained model
    if joblib is not None:
        st.subheader(" Save this model")
        buf = io.BytesIO()
        try:
            joblib.dump(pipe, buf)
            st.download_button(
                "Download trained pipeline (.joblib)",
                data=buf.getvalue(),
                file_name="trained_pipeline.joblib",
                mime="application/octet-stream"
            )
        except Exception:
            st.caption("Model download not available in this environment.")
    else:
        st.caption("joblib not available; skipping model download.")

elif uploaded_model:
    if joblib is None:
        st.error("joblib is not available, so the model cannot be loaded here.")
        st.stop()
    try:
        pipe = joblib.load(io.BytesIO(uploaded_model.getvalue()))
        st.success("Loaded model pipeline ")
        st.info("Upload a CSV too if you want SHAP explanations, since SHAP needs data to explain.")
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()


# ----------------------------
# SHAP Section
# ----------------------------
st.divider()
st.subheader(" SHAP explainability")

if pipe is None:
    st.info("Train or load a model pipeline to compute SHAP.")
    st.stop()

if df is None:
    st.info("Upload a CSV so the app has data to explain with SHAP.")
    st.stop()

# Re-derive X/y for explaining if we trained above
target_col_infer = None
if "Select target column" in st.session_state:
    # Streamlit doesn't store that label; we rely on actual df usage above.
    pass

# If user trained in this run, we already have X_train/X_test
if X_train is None or X_test is None:
    st.warning("Model was loaded, not trained here. Please re-upload data and re-train (recommended) for consistent preprocessing.")
    st.stop()

X_bg = _safe_sample(X_train, shap_bg_rows, random_state)
X_exp = _safe_sample(X_test, shap_rows, random_state)

c1, c2 = st.columns([1, 1])
c1.write(f"**Background rows:** {len(X_bg)}")
c2.write(f"**Explained rows:** {len(X_exp)}")

with st.spinner("Computing SHAP values (tree explainer)..."):
    t0 = time.time()
    try:
        feature_names, shap_values, X_exp_transformed = _compute_shap_for_tree_model(pipe, X_bg, X_exp)
        elapsed = time.time() - t0
        st.success(f"SHAP computed  ({elapsed:.2f}s)")
    except Exception as e:
        st.error("SHAP computation failed. If your model isn't tree-based, TreeExplainer may not work.")
        st.exception(e)
        st.stop()

tab1, tab2 = st.tabs(["Summary (bar)", "Summary (beeswarm)"])

with tab1:
    fig = _plot_shap_summary(feature_names, shap_values, X_exp_transformed, max_display=max_display, plot_type="bar")
    st.pyplot(fig, clear_figure=True)

with tab2:
    fig = _plot_shap_summary(feature_names, shap_values, X_exp_transformed, max_display=max_display, plot_type="dot")
    st.pyplot(fig, clear_figure=True)


# ----------------------------
# “Future phase” text (better framed)
# ----------------------------
st.divider()
st.subheader(" What I’ll build next (future phase)")

roadmap_map = {
    "Data pipeline": [
        "Add a data validation layer (schema checks, missing-value reports, outlier flags).",
        "Automate ingestion (scheduled pulls or API connectors), so uploads become optional.",
        "Version datasets so results are reproducible and comparable over time.",
    ],
    "Modeling": [
        "Offer model selection (XGBoost/LightGBM if available, plus linear baselines).",
        "Add hyperparameter tuning with guardrails (time limits + cross-validation).",
        "Support multiple targets and multi-output forecasting where relevant.",
    ],
    "Explainability": [
        "Add per-row explanations (waterfall/force plots) with a clean UI selector.",
        "Group features into human-friendly categories (e.g., climate, soil, market).",
        "Generate an exportable explanation report (PDF/HTML) for stakeholders.",
    ],
    "Deployment": [
        "Introduce a saved ‘model registry’ page inside the app (load, compare, promote).",
        "Add role-based access (viewer vs analyst) if this becomes multi-user.",
        "Containerize and pin versions to reduce “works locally” surprises.",
    ],
    "Monitoring": [
        "Track drift (feature drift + prediction drift) and alert on threshold breaches.",
        "Log SHAP distributions over time to detect explanation drift too.",
        "Add a simple evaluation dashboard for new data batches.",
    ],
}

st.markdown(
    """
**Right now**, this dashboard focuses on a stable baseline:
- Load data
- Train a reliable model
- Explain behavior with SHAP

**Next**, I’ll expand it in a structured way so each phase adds real value without breaking the previous one.
"""
)

st.markdown(f"### Focus area: {roadmap_focus}")
for item in roadmap_map[roadmap_focus]:
    st.write(f"• {item}")

st.markdown(
    """
If you want, I can also help you split this into a clean repo structure later:
`/src` for logic, `/app.py` thin UI layer, plus `/assets` and `/models`.
That usually makes Streamlit deployments calmer and less… dramatic. 
"""
)
