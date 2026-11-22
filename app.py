import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Movie Hit or Flop Predictor", page_icon="🎬", layout="centered")


@st.cache_resource
def load_model():
    with open("best_movie_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_options():
    with open("model_options.json") as f:
        return json.load(f)


model = load_model()
opt = load_options()

st.title("🎬 Movie Hit or Flop Predictor")
st.write(
    "Enter the details of a movie and the model will predict whether it will be a "
    "**HIT** (earns at least 1.5x its budget) or a **FLOP**."
)

col1, col2 = st.columns(2)

with col1:
    budget = st.number_input("Budget (USD)", min_value=100000, max_value=500000000,
                             value=50000000, step=1000000)
    runtime = st.slider("Runtime (minutes)", opt["runtime_min"], opt["runtime_max"], 110)
    year = st.slider("Release year", opt["year_min"], 2030, min(2024, 2030))
    score = st.slider("Expected IMDb score", float(opt["score_min"]), float(opt["score_max"]),
                      6.5, step=0.1)
    votes = st.number_input("Expected IMDb votes", min_value=1000, max_value=3000000,
                            value=100000, step=5000)

with col2:
    release_month = st.selectbox("Release month", opt["release_month"])
    genre = st.selectbox("Genre", opt["genre_grp"])
    rating = st.selectbox("MPAA rating", opt["rating_grp"])
    country = st.selectbox("Country", opt["country_grp"])
    company = st.selectbox("Production company", opt["company_grp"])
    director = st.selectbox("Director", opt["director_grp"])
    star = st.selectbox("Lead star", opt["star_grp"])

if st.button("Predict", type="primary", use_container_width=True):
    row = pd.DataFrame([{
        "budget": budget,
        "log_budget": np.log1p(budget),
        "runtime": runtime,
        "budget_per_min": budget / runtime,
        "year": year,
        "score": score,
        "votes": votes,
        "log_votes": np.log1p(votes),
        "release_month": release_month,
        "genre_grp": genre,
        "rating_grp": rating,
        "country_grp": country,
        "company_grp": company,
        "director_grp": director,
        "star_grp": star,
    }])

    prob = model.predict_proba(row)[0][1]
    pred = model.predict(row)[0]

    st.divider()
    if pred == 1:
        st.success(f"🎉 Predicted outcome: HIT  (Hit probability: {prob*100:.1f}%)")
    else:
        st.error(f"📉 Predicted outcome: FLOP  (Hit probability: {prob*100:.1f}%)")
    st.progress(float(prob))

st.caption("Model: Tuned Gradient Boosting (test ROC-AUC ≈ 0.87). "
           "Predictions are estimates based on historical data, not guarantees.")
