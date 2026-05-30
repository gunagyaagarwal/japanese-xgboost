# Anime Product Price ML App

This is an end-to-end learning project using Streamlit, pandas, NumPy, scikit-learn,
and XGBoost. It reads the root `data.csv`, converts the product array into a
dataframe, trains an XGBoost regression model, predicts product prices, and
ranks popular products for a selected user profile.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## What The Model Learns

The app predicts `price` from:

- `brand`
- `cat`
- `subcat`
- `stock`
- `source`
- `rating`
- `reviews`
- `tag_count`
- `name_length`

Because the dataset has 100 rows, use this as a hands-on learning app rather
than a highly accurate production model.

## Popular Product Prediction

The `Popular Products` tab takes:

- user preference
- age range
- location

The current CSV does not include real users, ages, locations, or purchase
history, so the app builds learning-friendly popularity labels from product
rating, reviews, stock, source, category, and preference keyword matches. Once
you collect real user interaction data, those generated labels can be replaced
with actual clicks, purchases, wishlists, or ratings.
