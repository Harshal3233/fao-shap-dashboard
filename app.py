import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="FAOSTAT XAI Case Study Dashboard", layout="wide")

# =========================================================
# Data
# =========================================================
@st.cache_data
def load_data():
    def load_country(path: str, country: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["Country"] = country
        return df

    df = pd.concat(
        [
            load_country("data/italy.csv", "Italy"),
            load_country("data/france.csv", "France"),
            load_country("data/germany.csv", "Germany"),
            load_country("data/spain.csv", "Spain"),
        ],
        ignore_index=True,
    )

    # Keep only what we need for this case study
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])
    df = df[["Country", "Item", "Value"]].copy()

    df["Country"] = df["Country"].astype(str)
    df["Item"] = df["Item"].astype(str)

    return df


df = load_data()
FEATURES = ["Country", "Item"]
TARGET = "Value"

# =========================================================
# Model training
# =========================================================
@st.cache_resource
def train_model(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)]
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return pipeline, X_train, X_test, y_train, y_test, mae, r2


pipeline, X_train, X_test, y_train, y_test, mae, r2 = train_model(df)

median_value = float(df["Value"].median()) if len(df) else np.nan
relative_mae = (mae / median_value) * 100 if median_value and not np.isnan(median_value) else np.nan

# =========================================================
# SHAP explainer (IMPORTANT: no caching with pipeline param)
# =========================================================
rf_model = pipeline.named_steps["model"]
explainer = shap.TreeExplainer(rf_model)


def transform_with_names(pipeline: Pipeline, X: pd.DataFrame):
    pre = pipeline.named_steps["preprocess"]
    X_trans = pre.transform(X).toarray()
    names = pre.named_transformers_["cat"].get_feature_names_out(FEATURES)
    return X_trans, names


# =========================================================
# Header
# =========================================================
st.title(" FAOSTAT Explainable ML Case Study (Italy • France • Germany • Spain)")

# =========================================================
# Full case study embedded in the dashboard
# =========================================================
with st.expander(" Full case study (what, why, how, results, interpretation)", expanded=True):
    st.markdown("""
## 1) Motivation and context
International organizations often work with **multi-country agricultural datasets** where the goal is not always forecasting,
but **interpretable comparison**: understanding differences and drivers across countries and commodities.

This dashboard is a **self-contained case study** using FAOSTAT-derived production data for four EU countries:
**Italy, France, Germany, Spain**.

## 2) Analytical questions
This project is structured around practical, policy-relevant questions:

1. **Crop vs Country:** How much of observed production magnitude is driven by *the crop (Item)* versus *country context*?
2. **Structural profiles:** Do countries exhibit distinct production patterns once crop composition is considered?
3. **Explainable ML:** Can machine learning be used as an **interpretability tool** to support transparent reasoning, not as a black-box predictor?

## 3) Data (FAOSTAT snapshot)
**Unit:** tonnes (t)

Each row represents a (Country, Item) observation with a production value.

Why a snapshot?
- A single-year cut keeps the case study focused on **cross-sectional structure** (country vs commodity),
  rather than time-series forecasting complexity.

## 4) Methodology overview
### 4.1 Why a Random Forest model?
A Random Forest regression model is used as a **flexible, non-parametric approximation** of the relationship between:
- Country (structural/institutional/geographic context)
- Item (commodity biology, agronomic characteristics)
- Production magnitude

The model’s purpose here is **not forecasting**. Instead, we treat it as a structured lens that can be explained.

### 4.2 Why SHAP?
High-cardinality categorical variables (many Items) and non-linear effects make “simple coefficients” less informative.
SHAP (SHapley Additive exPlanations) helps by:
- Breaking each prediction into **feature contributions**
- Keeping explanations additive (contributions sum to the prediction baseline)
- Allowing aggregation for **cross-country comparison** of structural drivers

Interpretation rules:
- **Positive SHAP** pushes the estimate upward
- **Negative SHAP** pushes the estimate downward

## 5) Model transparency and scale interpretation
We report MAE and R² for transparency. Because MAE in tonnes can feel abstract, we also show **Relative MAE**:

**Relative MAE (%) = MAE / median(Value) × 100**

Why median?
- Production values are often **skewed** (a few very large commodities),
  and the median better reflects a “typical” scale.

## 6) What the dashboard shows
### Explore
- Browse country-specific production composition
- Identify top contributing Items by total volume

### Predict + Explain
- Pick a (Country, Item) pair
- View a model estimate and a SHAP waterfall explanation

### Insights
- Aggregate SHAP values to compare:
  - average absolute **country effects**
  - average absolute **item effects**
- This addresses the central “Crop vs Country” question quantitatively

## 7) Key findings (in a nutshell)
- **Item type (crop)** is typically the strongest driver of production magnitude across observations.
- **Country effects** are present but secondary, reflecting structural and geographic differences.
- Countries can differ not only in *what they produce*, but in *how strongly item choice translates into production magnitude*.

## 8) Conclusion
This case study demonstrates how **explainable machine learning** can complement traditional descriptive analysis by making
cross-country comparison more transparent.

Rather than optimizing predictive performance, the workflow emphasizes:
- interpretability
- comparability
- traceable reasoning

This is the kind of approach that can support exploratory analysis, evidence synthesis, and communication of insights in
multi-country agricultural contexts.
    """)

# =========================================================
# Metrics row
# =========================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df)}")
c2.metric("MAE (tonnes)", f"{mae:,.0f}")
c3.metric("Relative MAE", f"{relative_mae:.1f}%" if not np.isnan(relative_mae) else "NA")
c4.metric("R²", f"{r2:.3f}")

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs(["Explore", "Predict + Explain", "Insights"])

# ----------------------------
# TAB 1: Explore
# ----------------------------
with tab1:
    st.caption("Browse and compare agricultural production by country and item.")

    country = st.selectbox("Country", sorted(df["Country"].unique()))
    df_c = df[df["Country"] == country].copy()

    left, right = st.columns([2, 1])
    with left:
        st.dataframe(df_c.sort_values("Value", ascending=False).head(30), use_container_width=True)
    with right:
        st.markdown("#### Quick stats")
        st.write(f"Items: **{df_c['Item'].nunique()}**")
        st.write(f"Total production: **{df_c['Value'].sum():,.0f} t**")
        st.write(f"Median item production: **{df_c['Value'].median():,.0f} t**")

    topn = st.slider("Top-N items (by total value)", 5, 30, 10)
    top_items = (
        df_c.groupby("Item", as_index=False)["Value"]
        .sum()
        .sort_values("Value", ascending=False)
        .head(topn)
    )
    fig = px.bar(top_items, x="Item", y="Value", title=f"{country}: Top {topn} items")
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "⬇️ Download filtered country data (CSV)",
        df_c.to_csv(index=False).encode("utf-8"),
        file_name=f"{country}_data.csv",
        mime="text/csv",
    )

# ----------------------------
# TAB 2: Predict + Explain
# ----------------------------
with tab2:
    st.caption("Select a country and item to view a model estimate and SHAP explanation.")

    cA, cB = st.columns(2)
    with cA:
        sel_country = st.selectbox("Country", sorted(df["Country"].unique()), key="p_country")
    with cB:
        # Only show items that exist for that country in your dataset
        items_for_country = sorted(df[df["Country"] == sel_country]["Item"].unique())
        sel_item = st.selectbox("Item", items_for_country, key="p_item")

    row = pd.DataFrame([{"Country": sel_country, "Item": sel_item}])
    pred = float(pipeline.predict(row)[0])

    st.metric("Model estimate (tonnes)", f"{pred:,.0f}")

    # SHAP for this single row
    X_trans, names = transform_with_names(pipeline, row)
    shap_vals = explainer.shap_values(X_trans, check_additivity=False)[0]

    exp = shap.Explanation(
        values=shap_vals,
        base_values=explainer.expected_value,
        feature_names=names,
    )

    st.markdown("#### SHAP explanation (waterfall)")
    plt.figure()
    shap.waterfall_plot(exp, max_display=12, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    st.markdown("#### Top SHAP contributors")
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:15]
    contrib = pd.DataFrame(
        {
            "feature": np.array(names)[top_idx],
            "shap_value": shap_vals[top_idx],
            "abs_shap": np.abs(shap_vals[top_idx]),
        }
    ).sort_values("abs_shap", ascending=False)[["feature", "shap_value"]]

    st.dataframe(contrib, use_container_width=True)

    st.download_button(
        "⬇️ Download SHAP contributors (CSV)",
        contrib.to_csv(index=False).encode("utf-8"),
        file_name=f"shap_contrib_{sel_country}_{sel_item}.csv".replace(" ", "_"),
        mime="text/csv",
    )

# ----------------------------
# TAB 3: Insights
# ----------------------------
with tab3:
    st.caption("Aggregate SHAP metrics to compare structural drivers across countries.")

    st.markdown("### Aggregated explainability: Crop vs Country")

    # Sample a manageable size for SHAP (fast + stable)
    sample_n = st.slider("Sample size for aggregated SHAP", 20, min(120, len(X_test)), min(60, len(X_test)))
    X_sample = X_test.sample(sample_n, random_state=42)

    X_trans, names = transform_with_names(pipeline, X_sample)
    shap_matrix = explainer.shap_values(X_trans, check_additivity=False)

    shap_df = pd.DataFrame(shap_matrix, columns=names)
    shap_df["Country"] = X_sample["Country"].values

    country_cols = [c for c in names if c.startswith("Country_")]
    item_cols = [c for c in names if c.startswith("Item_")]

    summary = (
        shap_df.groupby("Country")
        .apply(
            lambda g: pd.Series(
                {
                    "avg_abs_country_effect": g[country_cols].abs().mean().mean() if country_cols else 0.0,
                    "avg_abs_item_effect": g[item_cols].abs().mean().mean() if item_cols else 0.0,
                }
            )
        )
        .reset_index()
        .sort_values("avg_abs_item_effect", ascending=False)
    )

    st.dataframe(summary, use_container_width=True)

    melted = summary.melt(
        id_vars="Country",
        var_name="effect_type",
        value_name="avg_abs_shap",
    )
    fig2 = px.bar(
        melted,
        x="Country",
        y="avg_abs_shap",
        color="effect_type",
        barmode="group",
        title="Average absolute SHAP effect: Item vs Country",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Cross-country production composition (top items heatmap)")
    top_items_global = (
        df.groupby("Item", as_index=False)["Value"]
        .sum()
        .sort_values("Value", ascending=False)
        .head(20)["Item"]
        .tolist()
    )

    heat = (
        df[df["Item"].isin(top_items_global)]
        .groupby(["Item", "Country"], as_index=False)["Value"]
        .sum()
    )
    pivot = heat.pivot(index="Item", columns="Country", values="Value").fillna(0)

    # Table + optional heatmap chart
    st.dataframe(pivot, use_container_width=True)

    try:
        fig_h = px.imshow(pivot, aspect="auto", title="Top 20 items: Total production by country")
        st.plotly_chart(fig_h, use_container_width=True)
    except Exception:
        st.info("Heatmap rendering skipped (table above is available).")
