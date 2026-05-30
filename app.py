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
DEFAULT_AGE_RANGES = ["6-12", "8-14", "13-17", "18-25", "26-40"]
LOCATIONS = ["Delhi NCR", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune"]
PREFERENCES = [
    "Anime",
    "Apparel",
    "Collectibles",
    "Fashion",
    "Figures",
    "Food",
    "Gaming",
    "Gifting",
    "Home Decor",
    "Model Kits",
    "Posters",
    "Stationery",
    "Trading Cards",
]

PREFERENCE_KEYWORDS = {
    "Anime": ["anime", "manga", "naruto", "one piece", "demon slayer", "pokemon"],
    "Apparel": ["apparel", "clothing", "hoodie", "tee", "jacket", "shorts", "bag"],
    "Collectibles": ["figure", "figurine", "collectible", "tcg", "cards", "statue"],
    "Fashion": ["hoodie", "t-shirt", "shirt", "wear", "cosplay", "apparel"],
    "Figures": ["figure", "nendoroid", "figma", "figuarts", "scale", "display"],
    "Food": ["food", "ramen", "noodles", "pocky", "kitkat", "candy", "snack"],
    "Gaming": ["game", "gaming", "pokemon", "tcg", "cards", "play"],
    "Gifting": ["gift", "personalized", "custom", "keychain", "plush", "poster"],
    "Home Decor": ["poster", "lamp", "decor", "wall", "room", "scroll"],
    "Model Kits": ["gunpla", "gundam", "model kit", "hg", "mg", "rg", "hobby"],
    "Posters": ["poster", "print", "wall art", "room decor", "art print"],
    "Stationery": ["notebook", "pen", "stationery", "journal", "sticker", "paper"],
    "Trading Cards": ["tcg", "trading cards", "booster", "deck", "sleeves", "cards"],
}

AGE_CATEGORY_BOOSTS = {
    "13-17": ["Trading Cards", "Stationery", "Accessories"],
    "18-24": ["Fashion", "Collectibles", "Gaming"],
    "25-34": ["Collectibles", "Home Decor", "Fashion"],
    "35-44": ["Home Decor", "Collectibles", "Gifting"],
    "45+": ["Home Decor", "Gifting", "Stationery"],
}

AGE_RANGE_LABELS = {
    "6-12": "Kids",
    "8-14": "Kids",
    "13-17": "Teens",
    "18-25": "Young Adults",
    "26-40": "Adults",
}

LOCATION_SOURCE_BOOSTS = {
    "Delhi NCR": ["Amazon India", "Flipkart", "Etsy India"],
    "Mumbai": ["Amazon India", "Etsy India"],
    "Bengaluru": ["Amazon India", "Flipkart"],
    "Hyderabad": ["Flipkart", "Amazon India"],
    "Chennai": ["Amazon India", "Etsy India"],
    "Kolkata": ["Flipkart", "Etsy India"],
    "Pune": ["Amazon India", "Flipkart"],
}


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


def infer_age_group(product: pd.Series) -> dict:
    existing = product.get("ageGroup")
    if isinstance(existing, dict) and existing.get("ageRange"):
        return {
            "primary": existing.get("primary", "Unknown"),
            "ageRange": existing.get("ageRange", "Unknown"),
            "suitable": existing.get("suitable", []),
        }

    text = " ".join(
        [
            str(product.get("name", "")),
            str(product.get("cat", "")),
            str(product.get("subcat", "")),
            " ".join(product.get("tags", ()) or ()),
        ]
    ).lower()
    category = str(product.get("cat", ""))
    price = float(product.get("price", 0) or 0)

    if any(keyword in text for keyword in ["kids", "pocky", "candy", "pikachu", "cup noodle"]):
        return {"primary": "Kids", "ageRange": "6-12", "suitable": ["Kids", "Teens"]}

    if category in ["Trading Cards", "Apparel", "Model Kits"] and price < 2500:
        return {"primary": "Teens", "ageRange": "13-17", "suitable": ["Teens", "Young Adults"]}

    if category in ["Figures", "Collectibles", "Posters", "Handcrafted"] or "collector" in text:
        return {
            "primary": "Young Adults",
            "ageRange": "18-25",
            "suitable": ["Young Adults", "Adults"],
        }

    if price >= 7000 or any(keyword in text for keyword in ["premium", "investment", "scale"]):
        return {"primary": "Adults", "ageRange": "26-40", "suitable": ["Young Adults", "Adults"]}

    return {
        "primary": "Young Adults",
        "ageRange": "18-25",
        "suitable": ["Teens", "Young Adults", "Adults"],
    }


def infer_price_verification(product: pd.Series) -> dict:
    existing = product.get("priceVerification")
    if isinstance(existing, dict) and existing.get("status"):
        return existing

    return {
        "status": "dataset_rate",
        "note": "No explicit verification object in CSV; using listed dataset price.",
    }


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

    inferred_age_groups = df.apply(infer_age_group, axis=1)
    inferred_price_verification = df.apply(infer_price_verification, axis=1)
    df["age_group_primary"] = inferred_age_groups.apply(lambda value: value["primary"])
    df["age_group_age_range"] = inferred_age_groups.apply(lambda value: value["ageRange"])
    df["age_group_suitable"] = inferred_age_groups.apply(lambda value: tuple(value["suitable"]))
    df["age_group_suitable_text"] = df["age_group_suitable"].apply(lambda values: ", ".join(values))
    df["price_verification_status"] = inferred_price_verification.apply(lambda value: value["status"])
    df["price_verification_note"] = inferred_price_verification.apply(lambda value: value["note"])

    return df.dropna(subset=["price", "rating", "reviews"]).reset_index(drop=True)


def get_age_ranges(df: pd.DataFrame) -> list[str]:
    dataset_ranges = sorted(
        value
        for value in df["age_group_age_range"].dropna().unique()
        if value and value != "Unknown"
    )
    return dataset_ranges or DEFAULT_AGE_RANGES


def make_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "brand",
            "cat",
            "subcat",
            "stock",
            "source",
            "age_group_primary",
            "age_group_age_range",
            "price_verification_status",
            "rating",
            "reviews",
            "tag_count",
            "name_length",
        ]
    ].copy()


def preference_match_score(row: pd.Series, preference: str) -> float:
    searchable_text = " ".join(
        [
            str(row["name"]),
            str(row["brand"]),
            str(row["cat"]),
            str(row["subcat"]),
            str(row["tags_text"]),
        ]
    ).lower()
    keywords = PREFERENCE_KEYWORDS.get(preference, [])
    matches = sum(keyword in searchable_text for keyword in keywords)
    return min(matches / 3, 1.0)


def age_match_score(row: pd.Series, age_range: str) -> float:
    if row.get("age_group_age_range") == age_range:
        return 1.0

    age_label = AGE_RANGE_LABELS.get(age_range, "")
    suitable_groups = row.get("age_group_suitable", ())
    if age_label and age_label in suitable_groups:
        return 0.75

    if age_label and row.get("age_group_primary") == age_label:
        return 0.7

    category_boosts = AGE_CATEGORY_BOOSTS.get(age_range, [])
    if row.get("cat") in category_boosts:
        return 0.35

    return 0.0


def calculate_popularity_score(
    row: pd.Series,
    preference: str,
    age_range: str,
    location: str,
) -> float:
    rating_component = (row["rating"] / 5) * 35
    review_component = min(np.log1p(row["reviews"]) / np.log1p(5000), 1) * 25
    preference_component = preference_match_score(row, preference) * 25
    age_component = age_match_score(row, age_range) * 14
    location_component = 5 if row["source"] in LOCATION_SOURCE_BOOSTS.get(location, []) else 0
    stock_component = 4 if row["stock"] == "In Stock" else 1

    raw_score = (
        rating_component
        + review_component
        + preference_component
        + age_component
        + location_component
        + stock_component
    )
    return min(raw_score, 100)


def make_popularity_training_data(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    age_ranges = get_age_ranges(df)
    for preference in PREFERENCES:
        for age_range in age_ranges:
            for location in LOCATIONS:
                for _, product in df.iterrows():
                    rows.append(
                        {
                            "preference": preference,
                            "age_range": age_range,
                            "location": location,
                            "brand": product["brand"],
                            "cat": product["cat"],
                            "subcat": product["subcat"],
                            "stock": product["stock"],
                            "source": product["source"],
                            "age_group_primary": product["age_group_primary"],
                            "age_group_age_range": product["age_group_age_range"],
                            "price_verification_status": product["price_verification_status"],
                            "price": product["price"],
                            "rating": product["rating"],
                            "reviews": product["reviews"],
                            "tag_count": product["tag_count"],
                            "name_length": product["name_length"],
                            "popularity_score": calculate_popularity_score(
                                product,
                                preference,
                                age_range,
                                location,
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def make_popularity_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        [
            "preference",
            "age_range",
            "location",
            "brand",
            "cat",
            "subcat",
            "stock",
            "source",
            "age_group_primary",
            "age_group_age_range",
            "price_verification_status",
            "price",
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

    categorical_features = [
        "brand",
        "cat",
        "subcat",
        "stock",
        "source",
        "age_group_primary",
        "age_group_age_range",
        "price_verification_status",
    ]
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


@st.cache_resource
def train_popularity_model(df: pd.DataFrame):
    training_df = make_popularity_training_data(df)
    features = make_popularity_feature_frame(training_df)
    target = training_df["popularity_score"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    categorical_features = [
        "preference",
        "age_range",
        "location",
        "brand",
        "cat",
        "subcat",
        "stock",
        "source",
        "age_group_primary",
        "age_group_age_range",
        "price_verification_status",
    ]
    numeric_features = ["price", "rating", "reviews", "tag_count", "name_length"]

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
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
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
        "accuracy_percent": max(r2_score(y_test, predictions), 0) * 100,
    }
    return pipeline, metrics


def recommend_products(
    df: pd.DataFrame,
    pipeline: Pipeline,
    preference: str,
    age_range: str,
    location: str,
    limit: int,
) -> pd.DataFrame:
    candidates = df.copy()
    candidates["preference"] = preference
    candidates["age_range"] = age_range
    candidates["location"] = location

    features = make_popularity_feature_frame(candidates)
    candidates["predicted_popularity"] = np.clip(pipeline.predict(features), 0, 100)

    return candidates.sort_values(
        ["predicted_popularity", "rating", "reviews"],
        ascending=False,
    ).head(limit)


def format_rupees(value: float) -> str:
    return f"Rs. {value:,.0f}"


def show_dataset(df: pd.DataFrame) -> None:
    st.subheader("Dataset")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Products", len(df))
    metric_cols[1].metric("Average price", format_rupees(df["price"].mean()))
    metric_cols[2].metric("Average rating", f"{df['rating'].mean():.2f}")
    metric_cols[3].metric("Total reviews", f"{int(df['reviews'].sum()):,}")
    metric_cols[4].metric("Age ranges", len(get_age_ranges(df)))

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
                "age_group_primary",
                "age_group_age_range",
                "price_verification_status",
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
        metric_cols = st.columns(5)
        metric_cols[0].metric("Train rows", metrics["train_rows"])
        metric_cols[1].metric("Test rows", metrics["test_rows"])
        metric_cols[2].metric("MAE", format_rupees(metrics["mae"]))
        metric_cols[3].metric("R2 score", f"{metrics['r2']:.3f}")
        metric_cols[4].metric("Model accuracy", f"{max(metrics['r2'], 0) * 100:.1f}%")

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


def show_recommendations(df: pd.DataFrame) -> None:
    st.subheader("Popular Products By User Profile")
    st.caption("Choose a user profile and XGBoost ranks the products by predicted popularity.")

    profile_cols = st.columns(4)
    with profile_cols[0]:
        preference = st.selectbox("User preference", PREFERENCES)
    with profile_cols[1]:
        age_range = st.selectbox("Age range", get_age_ranges(df))
    with profile_cols[2]:
        location = st.selectbox("Location", LOCATIONS)
    with profile_cols[3]:
        limit = st.slider("Products", min_value=5, max_value=20, value=10)

    popularity_model, popularity_metrics = train_popularity_model(df)
    recommendations = recommend_products(
        df,
        popularity_model,
        preference,
        age_range,
        location,
        limit,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Top score", f"{recommendations['predicted_popularity'].max():.1f}/100")
    metric_cols[1].metric("Average rating", f"{recommendations['rating'].mean():.2f}")
    metric_cols[2].metric("Average price", format_rupees(recommendations["price"].mean()))
    metric_cols[3].metric("Model accuracy", f"{popularity_metrics['accuracy_percent']:.1f}%")

    st.dataframe(
        recommendations[
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
                "age_group_primary",
                "age_group_age_range",
                "price_verification_status",
                "predicted_popularity",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def show_learning_notes() -> None:
    st.subheader("What This App Is Doing")
    st.markdown(
        """
        1. Loads the product data into a pandas dataframe.
        2. Creates simple numeric features such as tag count and product-name length.
        3. One-hot encodes categorical columns like brand, category, stock, and source.
        4. Trains an XGBoost regression model to predict product price.
        5. Trains a second XGBoost model that ranks popular products from preference,
           verified product age group, location, and product details.
        6. Shows model accuracy/performance metrics, then lets you create product and user profiles.

        This dataset has only 100 rows, so treat the score as a learning signal rather than a
        production-grade benchmark. The popularity model uses the dataset age-group annotations,
        product quality signals, and generated profile labels because the CSV still does not
        include real user purchase or click history.
        """
    )


def main() -> None:
    st.title("Anime Product Price Prediction")
    st.write("A small Streamlit + pandas + NumPy + XGBoost project for learning ML end to end.")

    df = load_data()
    tab_data, tab_train, tab_predict, tab_popular, tab_notes = st.tabs(
        ["Data", "Train", "Predict Price", "Popular Products", "Learning Notes"]
    )

    with tab_data:
        show_dataset(df)

    with tab_train:
        pipeline, _ = show_training(df)

    with tab_predict:
        default_pipeline, _, _ = train_model(df, 160, 4, 0.08)
        show_prediction(df, default_pipeline)

    with tab_popular:
        show_recommendations(df)

    with tab_notes:
        show_learning_notes()


if __name__ == "__main__":
    main()
