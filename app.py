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
        n_estimators=140,
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
# Full embedded case study
# =========================================================
with st.expander(" Full case study (what, why, how, results, interpretation)", expanded=True):
    st.markdown("""
## 1) Motivation
In multi-country agricultural analysis, the goal is often **interpretable comparison** rather than forecasting.
This dashboard is designed to show how explainable ML can help answer: **what differs across countries, and why**.

## 2) Analytical questions (what this project answers)
1. **Crop vs Country:** Are differences in production mostly explained by the **crop (Item)** or the **country context**?
2. **Structural profiles:** Do countries show different production structure once crop composition is considered?
3. **Explainable ML:** Can ML be used as a transparent analytical tool (not a black box) to support reasoning?

## 3) Data
FAOSTAT-derived production values (unit: **tonnes**) for four EU countries.
Each row represents a (Country, Item) observation with an associated production value.

## 4) Why Random Forest?
Random Forest is used as a flexible approximation of the relationship between:
- **Country** (structural/institutional/geographic context)
- **Item** (commodity characteristics)
- **Production value**

Here, the model is primarily an **interpretability engine**: it learns patterns that we then explain.

## 5) Why SHAP?
SHAP explains predictions by decomposing them into additive contributions:
- **Positive SHAP** pushes the estimate up
- **Negative SHAP** pushes it down

This allows both:
- **Local explanations**: why a specific (country, item) has a certain estimate
- **Global/aggregated insights**: whether variation is mainly **crop-driven** or **country-driven**

## 6) How to read the results in this dashboard
- Explore: composition of production within a country, plus country-vs-country comparison
- Predict + Explain: model estimate + SHAP waterfall + top contributors
- Insights: aggregated SHAP magnitudes (strength of influence) and composition heatmap

## 7) Key findings (typical interpretation)
- **Item effects** often dominate: crop choice strongly influences magnitude.
- **Country effects** remain visible: structural differences shift production up/down.
- The dashboard enables quantitative comparison of “how crop-driven vs country-driven” each country appears.
    """)

# =========================================================
# Metrics row
# =========================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df)}")
c2.metric("MAE (tonnes)", f"{mae:,.0f}")
c3.metric("Relative MAE", f"{relative_mae:.1f}%" if not np.isnan(relative_mae) else "NA")
c4.metric("R²", f"{r2:.3f}")

with st.expander("How to read these metrics", expanded=False):
    st.markdown(f"""
- **MAE (tonnes)**: average absolute error on the held-out test split.
  If MAE is 200,000, the model is off by ~200k tonnes on average.

- **Relative MAE (%)**: MAE scaled by the **median** production value.
  Here it is **≈ {relative_mae:.1f}%** (median is used because production values are skewed).

- **R²**: variance explained. With only (Country, Item) as features, R² can be modest because real drivers like
  yield, harvested area, weather, and management are not included. That’s expected in this minimal case study.
    """)

# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3 = st.tabs(["Explore", "Predict + Explain", "Insights"])

# ----------------------------
# TAB 1: Explore
# ----------------------------
with tab1:
    st.subheader("Explore country production patterns")
    st.caption("Goal: understand production composition by item, then compare structures across countries.")

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

    topn = st.slider("Top-N items", 5, 30, 10)

    view_mode = st.radio(
        "Bar chart view",
        ["Total production (tonnes)", "Share of country total (%)"],
        horizontal=True
    )
    use_log = st.checkbox("Use log scale (for tonnes)", value=False)

    top_items = (
        df_c.groupby("Item", as_index=False)["Value"]
        .sum()
        .sort_values("Value", ascending=False)
        .head(topn)
    )

    if view_mode == "Share of country total (%)":
        denom = df_c["Value"].sum()
        top_items["Value"] = (top_items["Value"] / denom) * 100 if denom else 0
        y_label = "Share (%)"
    else:
        y_label = "Tonnes"

    fig = px.bar(
        top_items,
        x="Item",
        y="Value",
        title=f"{country}: Top {topn} items",
        labels={"Value": y_label}
    )
    if use_log and view_mode == "Total production (tonnes)":
        fig.update_yaxes(type="log")

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Interpretation: this bar chart shows the country’s production composition. "
        "Use Share (%) to compare countries fairly even if total output differs."
    )

    st.download_button(
        " Download this country subset (CSV)",
        df_c.to_csv(index=False).encode("utf-8"),
        file_name=f"{country}_data.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Compare two countries (structure)")
    st.caption("This comparison answers: do two countries differ mainly by scale, or by composition?")

    colA, colB = st.columns(2)
    countries_sorted = sorted(df["Country"].unique())
    with colA:
        cA = st.selectbox("Country A", countries_sorted, index=0, key="cA")
    with colB:
        cB = st.selectbox("Country B", countries_sorted, index=1 if len(countries_sorted) > 1 else 0, key="cB")

    dfA = df[df["Country"] == cA].groupby("Item", as_index=False)["Value"].sum()
    dfB = df[df["Country"] == cB].groupby("Item", as_index=False)["Value"].sum()

    top_global = (
        pd.concat([dfA, dfB])
        .groupby("Item", as_index=False)["Value"].sum()
        .sort_values("Value", ascending=False)
        .head(15)["Item"]
    )

    dfA2 = dfA[dfA["Item"].isin(top_global)].assign(Country=cA)
    dfB2 = dfB[dfB["Item"].isin(top_global)].assign(Country=cB)
    cmp = pd.concat([dfA2, dfB2], ignore_index=True)

    cmp_mode = st.radio("Compare as", ["Tonnes", "Share (%)"], horizontal=True, key="cmp_mode")
    if cmp_mode == "Share (%)":
        cmp["Value"] = cmp.groupby("Country")["Value"].transform(lambda s: (s / s.sum() * 100) if s.sum() else 0)

    fig_cmp = px.bar(
        cmp,
        x="Item",
        y="Value",
        color="Country",
        barmode="group",
        title=f"{cA} vs {cB}: Top items comparison"
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

# ----------------------------
# TAB 2: Predict + Explain
# ----------------------------
with tab2:
    st.subheader("Predict + Explain (SHAP)")
    st.caption("Goal: explain one selected (Country, Item) estimate using SHAP contributions.")

    cA, cB = st.columns(2)
    with cA:
        sel_country = st.selectbox("Country", sorted(df["Country"].unique()), key="p_country")
    with cB:
        items_for_country = sorted(df[df["Country"] == sel_country]["Item"].unique())
        sel_item = st.selectbox("Item", items_for_country, key="p_item")

    row = pd.DataFrame([{"Country": sel_country, "Item": sel_item}])
    pred = float(pipeline.predict(row)[0])

    st.metric("Model estimate (tonnes)", f"{pred:,.0f}")

    with st.expander("How to read the SHAP waterfall", expanded=False):
        st.markdown("""
- The model starts from a **baseline** (average prediction over training data).
- It then adds/subtracts contributions from features (Country_* and Item_*).
- The final sum becomes the **predicted value**.

Positive bars push the estimate up; negative bars push it down.
        """)

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

    st.markdown("#### Top SHAP contributors (with direction)")
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:15]
    contrib = pd.DataFrame(
        {
            "feature": np.array(names)[top_idx],
            "shap_value": shap_vals[top_idx],
        }
    )
    contrib["direction"] = np.where(contrib["shap_value"] >= 0, "↑ increases", "↓ decreases")
    contrib = contrib.sort_values(by="shap_value", ascending=False)

    st.dataframe(contrib, use_container_width=True)

    st.download_button(
        " Download SHAP contributors (CSV)",
        contrib.to_csv(index=False).encode("utf-8"),
        file_name=f"shap_contrib_{sel_country}_{sel_item}.csv".replace(" ", "_"),
        mime="text/csv",
    )

# ----------------------------
# TAB 3: Insights
# ----------------------------
with tab3:
    st.subheader("Insights (aggregated explainability)")
    st.caption("Goal: compare the strength of crop-driven vs country-driven effects across countries.")

    with st.expander("What do these aggregated SHAP numbers mean?", expanded=True):
        st.markdown("""
These metrics summarize SHAP values over a sample of rows:

- **avg_abs_item_effect**: average absolute SHAP magnitude for Item_* features.
  It answers: *how strongly crop choice typically moves production estimates up/down?*

- **avg_abs_country_effect**: average absolute SHAP magnitude for Country_* features.
  It answers: *how strongly country context typically shifts production estimates up/down?*

Because we take **absolute values**, this measures **strength of influence**, not direction.

 Interpretation:
- If **avg_abs_item_effect > avg_abs_country_effect**, differences are mostly **crop-driven**.
- If **avg_abs_country_effect is high**, the **country context** has a stronger structural role.
        """)

    st.markdown("### A) Crop vs Country: aggregated SHAP comparison")

    sample_n = st.slider("Sample size used for aggregated SHAP (trade-off: speed vs stability)", 20, min(160, len(X_test)), min(80, len(X_test)))
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
    )
    summary["item_to_country_ratio"] = summary["avg_abs_item_effect"] / (summary["avg_abs_country_effect"] + 1e-9)
    summary = summary.sort_values("item_to_country_ratio", ascending=False)

    st.dataframe(summary, use_container_width=True)

    metric = st.selectbox(
        "Choose what to plot",
        ["avg_abs_item_effect", "avg_abs_country_effect", "item_to_country_ratio"]
    )

    fig_metric = px.bar(
        summary.sort_values(metric, ascending=False),
        x="Country",
        y=metric,
        title=f"Country comparison: {metric}",
    )
    st.plotly_chart(fig_metric, use_container_width=True)

    st.caption(
        "Tip: **item_to_country_ratio > 1** means item effects dominate; **< 1** means country effects dominate."
    )

    st.divider()
    st.markdown("### B) Cross-country composition: top items heatmap")
    st.caption("This section compares which items dominate total production across the four countries.")

    top_k = st.slider("How many top items to include", 10, 40, 20)
    top_items_global = (
        df.groupby("Item", as_index=False)["Value"]
        .sum()
        .sort_values("Value", ascending=False)
        .head(top_k)["Item"]
        .tolist()
    )

    heat = (
        df[df["Item"].isin(top_items_global)]
        .groupby(["Item", "Country"], as_index=False)["Value"]
        .sum()
    )
    pivot = heat.pivot(index="Item", columns="Country", values="Value").fillna(0)

    view = st.radio("Heatmap values", ["Tonnes", "Share within country (%)"], horizontal=True)
    pivot_view = pivot.copy()
    if view == "Share within country (%)":
        pivot_view = pivot_view.div(pivot_view.sum(axis=0).replace(0, np.nan), axis=1) * 100
        pivot_view = pivot_view.fillna(0)

    st.dataframe(pivot_view, use_container_width=True)

    try:
        fig_h = px.imshow(
            pivot_view,
            aspect="auto",
            title=f"Top {top_k} items: {view}",
        )
        st.plotly_chart(fig_h, use_container_width=True)
    except Exception:
        st.info("Heatmap rendering skipped (table above is available).")

    with st.expander("How to interpret this heatmap", expanded=False):
        st.markdown("""
- In **Tonnes** mode, you see absolute magnitude (big producers dominate visually).
- In **Share (%)** mode, you see structure: which crops make up a larger fraction of a country’s total.
This helps separate “big because total is big” from “big because composition is different”.
