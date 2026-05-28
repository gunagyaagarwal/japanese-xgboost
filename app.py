import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


DATA_PATH = Path(__file__).with_name("data.csv")
RANDOM_STATE = 42


st.set_page_config(
    page_title="Anime Product Price ML",
    page_icon="ML",
    layout="wide",
)


def _extract_products_array(raw_text: str) -> list[dict]:
    match = re.search(r"const\s+PRODUCTS\s*=\s*(\[.*\])\s*;", raw_text, re.DOTALL)
    if not match:
        raise ValueError("Could not find a `const PRODUCTS = [...]` array in data.csv.")
    return json.loads(match.group(1))


@st.cache_data
def load_data() -> pd.DataFrame:
    raw_text = DATA_PATH.read_text(encoding="utf-8", errors="replace")
    products = _extract_products_array(raw_text)
    df = pd.DataFrame(products)

    df["tags"] = df["tags"].apply(lambda value: tuple(value) if isinstance(value, list) else ())
    df["tags_text"] = df["tags"].apply(lambda tags: ", ".join(tags))
    df["tag_count"] = df["tags"].apply(len)
    df["name_length"] = df["name"].str.len()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["reviews"] = pd.to_numeric(df["reviews"], errors="coerce")

    return df.dropna(subset=["price", "rating", "reviews"]).reset_index(drop=True)


def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "brand",
            "cat",
            "subcat",
            "stock",
            "source",
            "rating",
            "reviews",
            "tag_count",
            "name_length",
        ]
    ].copy()


@st.cache_resource
def train_model(df: pd.DataFrame, n_estimators: int, max_depth: int, learning_rate: float):
    features = make_feature_frame(df)
    target = df["price"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    categorical_features = ["brand", "cat", "subcat", "stock", "source"]
    numeric_features = ["rating", "reviews", "tag_count", "name_length"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("numeric", "passthrough", numeric_features),
        ],
        remainder="drop",
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
    }

    return pipeline, metrics, x_test.assign(actual_price=y_test, predicted_price=predictions)


def format_rupees(value: float) -> str:
    return f"Rs. {value:,.0f}"


def show_dataset(df: pd.DataFrame) -> None:
    st.subheader("Dataset")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Products", len(df))
    metric_cols[1].metric("Average price", format_rupees(df["price"].mean()))
    metric_cols[2].metric("Average rating", f"{df['rating'].mean():.2f}")
    metric_cols[3].metric("Total reviews", f"{int(df['reviews'].sum()):,}")

    st.dataframe(
        df[
            [
                "name",
                "brand",
                "cat",
                "subcat",
                "price",
                "rating",
                "reviews",
                "stock",
                "source",
                "tags_text",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def show_training(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    st.subheader("Train XGBoost Regressor")
    left, right = st.columns([1, 2])

    with left:
        st.caption("Tune the core XGBoost settings and retrain the model.")
        n_estimators = st.slider("Trees", min_value=30, max_value=500, value=160, step=10)
        max_depth = st.slider("Max depth", min_value=2, max_value=10, value=4)
        learning_rate = st.slider(
            "Learning rate",
            min_value=0.01,
            max_value=0.5,
            value=0.08,
            step=0.01,
        )

    pipeline, metrics, results = train_model(df, n_estimators, max_depth, learning_rate)

    with right:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Train rows", metrics["train_rows"])
        metric_cols[1].metric("Test rows", metrics["test_rows"])
        metric_cols[2].metric("MAE", format_rupees(metrics["mae"]))
        metric_cols[3].metric("R2 score", f"{metrics['r2']:.3f}")

        chart_df = results[["actual_price", "predicted_price"]].reset_index(drop=True)
        st.line_chart(chart_df)

    st.dataframe(
        results.sort_values("actual_price", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    return pipeline, metrics


def show_prediction(df: pd.DataFrame, pipeline: Pipeline) -> None:
    st.subheader("Predict A Product Price")
    st.caption("Create a product profile and let the trained XGBoost model estimate its price.")

    input_cols = st.columns(3)
    with input_cols[0]:
        brand = st.selectbox("Brand", sorted(df["brand"].unique()))
        category = st.selectbox("Category", sorted(df["cat"].unique()))
        subcategory = st.selectbox("Subcategory", sorted(df["subcat"].unique()))
    with input_cols[1]:
        stock = st.selectbox("Stock", sorted(df["stock"].unique()))
        source = st.selectbox("Source", sorted(df["source"].unique()))
        rating = st.slider("Rating", min_value=1.0, max_value=5.0, value=4.6, step=0.1)
    with input_cols[2]:
        reviews = st.number_input("Reviews", min_value=0, value=750, step=50)
        tag_count = st.slider("Number of tags", min_value=0, max_value=15, value=6)
        name_length = st.slider("Product name length", min_value=5, max_value=120, value=42)

    sample = pd.DataFrame(
        [
            {
                "brand": brand,
                "cat": category,
                "subcat": subcategory,
                "stock": stock,
                "source": source,
                "rating": rating,
                "reviews": reviews,
                "tag_count": tag_count,
                "name_length": name_length,
            }
        ]
    )

    predicted_price = float(np.clip(pipeline.predict(sample)[0], a_min=0, a_max=None))
    st.metric("Predicted price", format_rupees(predicted_price))


def show_learning_notes() -> None:
    st.subheader("What This App Is Doing")
    st.markdown(
        """
        1. Loads the product data into a pandas dataframe.
        2. Creates simple numeric features such as tag count and product-name length.
        3. One-hot encodes categorical columns like brand, category, stock, and source.
        4. Trains an XGBoost regression model to predict product price.
        5. Shows test-set metrics, then lets you create a new product profile for prediction.

        This dataset has only 100 rows, so treat the score as a learning signal rather than a
        production-grade benchmark. More rows usually make the model much more reliable.
        """
    )


def main() -> None:
    st.title("Anime Product Price Prediction")
    st.write("A small Streamlit + pandas + NumPy + XGBoost project for learning ML end to end.")

    df = load_data()
    tab_data, tab_train, tab_predict, tab_notes = st.tabs(
        ["Data", "Train", "Predict", "Learning Notes"]
    )

    with tab_data:
        show_dataset(df)

    with tab_train:
        pipeline, _ = show_training(df)

    with tab_predict:
        default_pipeline, _, _ = train_model(df, 160, 4, 0.08)
        show_prediction(df, default_pipeline)

    with tab_notes:
        show_learning_notes()


if __name__ == "__main__":
    main()
