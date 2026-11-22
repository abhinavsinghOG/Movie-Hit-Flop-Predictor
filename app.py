import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CineScope | Hit or Flop Predictor",
                   page_icon="🎬", layout="wide")

st.markdown("""
<style>
.hero {padding: 2rem 1.5rem; border-radius: 16px; margin-bottom: 1.2rem;
       background: linear-gradient(135deg, #b20710 0%, #141414 75%);}
.hero h1 {margin: 0; font-size: 2.4rem;}
.hero p {margin: .4rem 0 0; opacity: .85;}
.result {padding: 1.6rem; border-radius: 16px; text-align: center; margin-top: 1rem;}
.hit {background: rgba(46,204,113,.15); border: 2px solid #2ecc71;}
.flop {background: rgba(231,76,60,.15); border: 2px solid #e74c3c;}
.result h2 {margin: 0; font-size: 2.4rem;}
.result p {margin: .4rem 0 0; font-size: 1.15rem;}
div.stButton > button {background: #E50914; color: white; border: none;
       border-radius: 10px; font-weight: 700; padding: .7rem;}
div.stButton > button:hover {background: #ff1f2b; color: white;}
</style>
""", unsafe_allow_html=True)


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

# ---------- Sidebar inputs ----------
st.sidebar.title("🎞️ Movie Details")
budget = st.sidebar.number_input("Budget (USD)", 100000, 500000000, 50000000, 1000000)
runtime = st.sidebar.slider("Runtime (minutes)", opt["runtime_min"], opt["runtime_max"], 110)
year = st.sidebar.slider("Release year", opt["year_min"], 2030, 2024)
score = st.sidebar.slider("Expected IMDb score", float(opt["score_min"]),
                          float(opt["score_max"]), 6.5, step=0.1)
votes = st.sidebar.number_input("Expected IMDb votes", 1000, 3000000, 100000, 5000)

st.sidebar.divider()
release_month = st.sidebar.selectbox("Release month", opt["release_month"])
genre = st.sidebar.selectbox("Genre", opt["genre_grp"])
rating = st.sidebar.selectbox("MPAA rating", opt["rating_grp"])
country = st.sidebar.selectbox("Country", opt["country_grp"])
company = st.sidebar.selectbox("Production company", opt["company_grp"])
director = st.sidebar.selectbox("Director", opt["director_grp"])
star = st.sidebar.selectbox("Lead star", opt["star_grp"])

# ---------- Main page ----------
st.markdown("""
<div class="hero">
  <h1>🎬 CineScope</h1>
  <p>Predict whether your movie will be a <b>HIT</b> or a <b>FLOP</b> at the box office.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔮 Prediction", "ℹ️ About"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Budget", f"${budget/1e6:,.1f}M")
    c2.metric("IMDb score", f"{score:.1f} ⭐")
    c3.metric("Runtime", f"{runtime} min")

    if st.button("Predict Box-Office Outcome", use_container_width=True):
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
        prob = float(model.predict_proba(row)[0][1])
        pred = int(model.predict(row)[0])

        if pred == 1:
            st.markdown(f"""<div class="result hit"><h2>🎉 HIT</h2>
                <p>Hit probability: <b>{prob*100:.1f}%</b></p></div>""",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result flop"><h2>📉 FLOP</h2>
                <p>Hit probability: <b>{prob*100:.1f}%</b></p></div>""",
                        unsafe_allow_html=True)
        st.progress(prob)
    else:
        st.info("Set the movie details in the sidebar, then click **Predict**.")

with tab2:
    st.markdown("""
**How it works**
- A movie is labelled a **HIT** if its gross is at least 1.5x its budget, otherwise a **FLOP**.
- The model is a tuned **Gradient Boosting** classifier (test ROC-AUC ≈ 0.87, accuracy ≈ 78%).
- Inputs: budget, runtime, year, expected IMDb score and votes, release month, genre,
  rating, country, company, director and star.

*Predictions are estimates based on historical data, not guarantees.*
""")
