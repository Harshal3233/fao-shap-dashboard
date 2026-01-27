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

st.set_page_config(page_title="FAOSTAT XAI Dashboard", layout="wide")

@st.cache_data
def load_data():
    def load_country(path, country):
        df = pd.read_csv(path)
        df["Country"] = country
        return df

    df = pd.concat([
        load_country("data/italy.csv", "Italy"),
        load_country("data/france.csv", "France"),
        load_country("data/germany.csv", "Germany"),
        load_country("data/spain.csv", "Spain"),
    ], ignore_index=True)

    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])

    # Single-year snapshot => minimal features
    df = df[["Country", "Item", "Value"]].copy()
    df["Country"] = df["Country"].astype(str)
    df["Item"] = df["Item"].astype(str)
    return df

df = load_data()
FEATURES = ["Country", "Item"]
TARGET = "Value"

@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES)])
    model = RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1)

    pipe = Pipeline([("preprocess", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return pipe, X_train, X_test, mae, r2

pipeline, X_train, X_test, mae, r2 = train_model(df)

rf = pipeline.named_steps["model"]
explainer = shap.TreeExplainer(rf)

def transform_with_names(pipeline, X):
    pre = pipeline.named_steps["preprocess"]
    X_trans = pre.transform(X).toarray()
    ohe = pre.named_transformers_["cat"]
    names = ohe.get_feature_names_out(FEATURES)
    return X_trans, names

st.title(" FAOSTAT Explainable AI Dashboard")
with st.expander(" About this project (context, data & methodology)", expanded=True):
    st.markdown("""
    ###  Project context
    This dashboard is a **demonstration data science project** inspired by the type of analytical
    work conducted at international organizations such as the FAO.

    It compares **agricultural production volumes** across four EU countries
    (Italy, France, Germany, Spain) using FAOSTAT-derived data.

    The goal is **not forecasting**, but **interpretation and comparison**:
    understanding what drives differences in production values across countries and items.

    ###  Data
    - Source: **FAOSTAT** (Crops and livestock products)
    - Countries: Italy, France, Germany, Spain
    - Scope: Single-year snapshot
    - Unit: Metric tonnes (t)
    - Each row represents: *(Country, Item → Production Value)*

    ###  Model
    A **Random Forest regression model** is trained to estimate production values using:
    - Country (categorical)
    - Item (crop type)

    Performance metrics (MAE, R²) are displayed for transparency, not as an official benchmark.

    ###  Explainability (SHAP)
    SHAP (SHapley Additive exPlanations) decomposes each prediction into contributions:
    - Positive SHAP values push the estimate upward
    - Negative SHAP values push the estimate downward

    Aggregated SHAP values support cross-country comparison of structural drivers.
    """)
    with st.expander("📘 About this project (summary, data, methodology & conclusions)", expanded=True):
    st.markdown("""
    ### 🌍 Project context
    This dashboard is a **demonstration data science project** inspired by analytical work
    carried out at international organizations such as the FAO.

    It compares **agricultural production volumes** across four EU countries
    (Italy, France, Germany, Spain) using FAOSTAT-derived data.

    The objective is **interpretation and comparison**, not forecasting.

    ### 📊 Data
    - Source: **FAOSTAT**
    - Countries: Italy, France, Germany, Spain
    - Scope: Single-year snapshot
    - Unit: Metric tonnes (t)

    ### 🤖 Model
    A **Random Forest regression model** estimates production values using:
    - Country
    - Item (crop type)

    Model metrics (MAE, R²) are shown for transparency.

    ###  Explainability (SHAP)
    SHAP explains each estimate by decomposing it into feature contributions.
    This enables transparent interpretation at both **individual** and **aggregate** levels.

    ###  Key findings (in a nutshell)
    - Item type is the strongest driver of production magnitude
    - Country effects reflect structural and geographic differences
    - Countries vary in how strongly item choice influences output

    ###  Conclusion
    Explainable machine learning can complement traditional statistics by
    revealing **structural drivers** behind observed agricultural patterns.
    """)

    st.caption(
        " Disclaimer: This dashboard is an educational case study and not an official FAO product."
    )

    st.caption(
        " Disclaimer: This dashboard is an educational case study. "
        "Results are not official FAO estimates or forecasts."
    )
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df)}")
c2.metric("MAE", f"{mae:,.0f}")
c3.metric("R²", f"{r2:.3f}")

tab1, tab2, tab3 = st.tabs(["Explore", "Predict + Explain", "Insights"])

with tab1:
    st.caption("Browse and compare agricultural production by country and item.")
    st.subheader("Explore")
    country = st.selectbox("Country", sorted(df["Country"].unique()))
    df_c = df[df["Country"] == country]
    st.dataframe(df_c.head(30), use_container_width=True)

    topn = st.slider("Top-N items", 5, 30, 10)
    top_items = df_c.groupby("Item", as_index=False)["Value"].sum().sort_values("Value", ascending=False).head(topn)
    st.plotly_chart(px.bar(top_items, x="Item", y="Value", title=f"{country}: Top {topn} items"), use_container_width=True)

    st.download_button(
        "⬇️ Download this country data (CSV)",
        df_c.to_csv(index=False).encode("utf-8"),
        file_name=f"{country}_data.csv",
        mime="text/csv",
    )

with tab2:
    st.caption("Select a country and item to view a model estimate and SHAP explanation.")
    st.subheader("Predict + Explain (SHAP)")
    country = st.selectbox("Country", sorted(df["Country"].unique()), key="p_country")
    item = st.selectbox("Item", sorted(df[df["Country"] == country]["Item"].unique()), key="p_item")

    row = pd.DataFrame([{"Country": country, "Item": item}])
    pred = pipeline.predict(row)[0]
    st.metric("Predicted Value", f"{pred:,.0f}")

    X_trans, names = transform_with_names(pipeline, row)
    sv = explainer.shap_values(X_trans, check_additivity=False)[0]
    exp = shap.Explanation(values=sv, base_values=explainer.expected_value, feature_names=names)

    plt.figure()
    shap.waterfall_plot(exp, max_display=12, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)

    top_idx = np.argsort(np.abs(sv))[::-1][:12]
    contrib = pd.DataFrame({"feature": np.array(names)[top_idx], "shap_value": sv[top_idx]})
    st.dataframe(contrib, use_container_width=True)

    st.download_button(
        "⬇️ Download SHAP contributors (CSV)",
        contrib.to_csv(index=False).encode("utf-8"),
        file_name=f"shap_{country}_{item}.csv".replace(" ", "_"),
        mime="text/csv",
    )

with tab3:
    st.caption("View aggregated SHAP metrics to compare structural drivers across countries.")
    st.subheader("Insights (aggregated SHAP)")
    X_sample = X_test.sample(min(30, len(X_test)), random_state=42)
    X_trans, names = transform_with_names(pipeline, X_sample)
    shap_vals = explainer.shap_values(X_trans, check_additivity=False)

    shap_df = pd.DataFrame(shap_vals, columns=names)
    shap_df["Country"] = X_sample["Country"].values

    country_cols = [c for c in names if c.startswith("Country_")]
    item_cols = [c for c in names if c.startswith("Item_")]

    summary = shap_df.groupby("Country").apply(
        lambda g: pd.Series({
            "avg_abs_country_effect": g[country_cols].abs().mean().mean(),
            "avg_abs_item_effect": g[item_cols].abs().mean().mean(),
        })
    ).reset_index()

    st.dataframe(summary, use_container_width=True)

    melted = summary.melt(id_vars="Country", var_name="effect_type", value_name="avg_abs_shap")
    st.plotly_chart(px.bar(melted, x="Country", y="avg_abs_shap", color="effect_type", barmode="group"), use_container_width=True)
