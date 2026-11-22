import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CineScope | Box-Office Predictor",
                   page_icon="🎬", layout="wide")

# ------------------------------------------------------------------ STYLE
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {font-family: 'Inter', sans-serif;}
#MainMenu, footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2rem; max-width: 1100px;}

.stApp {
  background:
    radial-gradient(1000px 500px at 85% -10%, rgba(139,92,246,.18), transparent 60%),
    radial-gradient(800px 400px at -10% 10%, rgba(244,63,94,.12), transparent 60%),
    #0B0F1A;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(18,24,41,.85);
  border-right: 1px solid rgba(255,255,255,.06);
}
section[data-testid="stSidebar"] h1 {font-size: 1.3rem; font-weight: 800;}

/* Inputs */
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
  border-radius: 12px !important;
  background: rgba(255,255,255,.04) !important;
  border: 1px solid rgba(255,255,255,.08) !important;
}

/* Hero */
.hero {
  position: relative; overflow: hidden;
  padding: 2.2rem 2rem; border-radius: 24px; margin-bottom: 1.4rem;
  background: linear-gradient(120deg, #8B5CF6 0%, #F43F5E 60%, #FB923C 100%);
  background-size: 200% 200%; animation: shift 10s ease infinite;
  box-shadow: 0 20px 60px rgba(139,92,246,.25);
}
.hero::after {
  content: "🎬"; position: absolute; right: 1.5rem; top: .6rem;
  font-size: 6rem; opacity: .18;
}
.hero h1 {margin: 0; font-size: 2.6rem; font-weight: 800; letter-spacing: -1px; color: #fff;}
.hero p {margin: .5rem 0 0; font-size: 1.05rem; color: rgba(255,255,255,.92); max-width: 560px;}
@keyframes shift {0%,100% {background-position: 0% 50%;} 50% {background-position: 100% 50%;}}

/* Stat cards */
.stats {display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 1rem;}
.stat {
  padding: 1.1rem 1.2rem; border-radius: 18px;
  background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
  backdrop-filter: blur(10px);
}
.stat small {color: #94A3B8; font-weight: 500; font-size: .8rem; text-transform: uppercase; letter-spacing: .06em;}
.stat div {font-size: 1.7rem; font-weight: 800; margin-top: .2rem; color: #F1F5F9;}

/* Chips */
.chips {display: flex; flex-wrap: wrap; gap: 8px; margin: .2rem 0 1.2rem;}
.chip {
  padding: .35rem .8rem; border-radius: 999px; font-size: .85rem; font-weight: 500;
  background: rgba(139,92,246,.14); border: 1px solid rgba(139,92,246,.35); color: #DDD6FE;
}

/* Button */
div.stButton > button {
  width: 100%; border: none; border-radius: 14px; padding: .85rem 1rem;
  font-weight: 700; font-size: 1.05rem; color: #fff;
  background: linear-gradient(90deg, #8B5CF6, #F43F5E);
  box-shadow: 0 10px 30px rgba(244,63,94,.3);
  transition: transform .15s ease, box-shadow .15s ease;
}
div.stButton > button:hover {
  transform: translateY(-2px); color: #fff;
  box-shadow: 0 14px 38px rgba(244,63,94,.45);
}

/* Result */
.result {
  display: flex; align-items: center; gap: 2rem; flex-wrap: wrap;
  padding: 1.8rem 2rem; border-radius: 24px; margin-top: 1.2rem;
  background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.1);
  animation: pop .45s ease;
}
@keyframes pop {from {opacity: 0; transform: translateY(12px);} to {opacity: 1; transform: none;}}
.ring {
  width: 170px; height: 170px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.ring-inner {
  width: 138px; height: 138px; border-radius: 50%; background: #0B0F1A;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.ring-inner span {font-size: 2.4rem; font-weight: 800; color: #fff;}
.ring-inner small {color: #94A3B8; font-size: .75rem; text-transform: uppercase; letter-spacing: .08em;}
.verdict h2 {margin: 0; font-size: 2.4rem; font-weight: 800; letter-spacing: -1px;}
.verdict p {margin: .4rem 0 0; color: #CBD5E1; font-size: 1.02rem;}
.badge {
  display: inline-block; margin-top: .8rem; padding: .3rem .8rem; border-radius: 999px;
  font-size: .8rem; font-weight: 600; background: rgba(255,255,255,.08); color: #E2E8F0;
}
.hit h2 {color: #34D399;}
.flop h2 {color: #FB7185;}

/* Tabs */
button[data-baseweb="tab"] {font-weight: 600;}
.note {color: #64748B; font-size: .85rem; margin-top: 1.2rem;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ DATA
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


def other_last(values):
    """Keep the list alphabetical but move 'Other' to the end."""
    return [v for v in values if v != "Other"] + (["Other"] if "Other" in values else [])


month_order = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December", "Unknown"]

# ------------------------------------------------------------------ SIDEBAR
st.sidebar.title("🎞️ Movie Details")
st.sidebar.caption("Fill in the details, then hit Predict.")

st.sidebar.subheader("Production")
budget = st.sidebar.number_input("Budget (USD)", 100000, 500000000, 50000000, 1000000)
runtime = st.sidebar.slider("Runtime (minutes)", opt["runtime_min"], 250, 110)
year = st.sidebar.slider("Release year", opt["year_min"], 2030, 2024)

st.sidebar.subheader("Audience")
score = st.sidebar.slider("Expected IMDb score", float(opt["score_min"]),
                          10.0, 6.5, step=0.1)
votes = st.sidebar.number_input("Expected IMDb votes", 1000, 3000000, 100000, 5000)

st.sidebar.subheader("Release & Cast")
months = [m for m in month_order if m in opt["release_month"]]
release_month = st.sidebar.selectbox("Release month", months)
genre = st.sidebar.selectbox("Genre", other_last(opt["genre_grp"]))
rating = st.sidebar.selectbox("MPAA rating", other_last(opt["rating_grp"]))
country = st.sidebar.selectbox("Country", other_last(opt["country_grp"]))
company = st.sidebar.selectbox("Production company", other_last(opt["company_grp"]))
director = st.sidebar.selectbox("Director", other_last(opt["director_grp"]))
star = st.sidebar.selectbox("Lead star", other_last(opt["star_grp"]))

# ------------------------------------------------------------------ MAIN
st.markdown(
    '<div class="hero"><h1>CineScope</h1>'
    '<p>AI-powered box-office predictor. See whether a movie is headed for '
    'a <b>HIT</b> or a <b>FLOP</b> in seconds.</p></div>',
    unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔮 Prediction", "ℹ️ About"])

with tab1:
    st.markdown(
        f'<div class="stats">'
        f'<div class="stat"><small>Budget</small><div>${budget/1e6:,.1f}M</div></div>'
        f'<div class="stat"><small>IMDb score</small><div>{score:.1f} ⭐</div></div>'
        f'<div class="stat"><small>Runtime</small><div>{runtime} min</div></div>'
        f'</div>',
        unsafe_allow_html=True)

    st.markdown(
        f'<div class="chips">'
        f'<span class="chip">🎭 {genre}</span>'
        f'<span class="chip">🔞 {rating}</span>'
        f'<span class="chip">📅 {release_month} {year}</span>'
        f'<span class="chip">🌍 {country}</span>'
        f'<span class="chip">🏢 {company}</span>'
        f'</div>',
        unsafe_allow_html=True)

    if st.button("Predict Box-Office Outcome"):
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

        gap = abs(prob - 0.5)
        if gap >= 0.30:
            confidence = "High confidence"
        elif gap >= 0.15:
            confidence = "Moderate confidence"
        else:
            confidence = "Close call"

        color = "#34D399" if pred == 1 else "#FB7185"
        cls = "hit" if pred == 1 else "flop"
        title = "🎉 Likely HIT" if pred == 1 else "📉 Likely FLOP"
        line = ("Expected to earn at least 1.5x its budget."
                if pred == 1 else
                "Expected to earn less than 1.5x its budget.")
        deg = prob * 360

        st.markdown(
            f'<div class="result {cls}">'
            f'<div class="ring" style="background: conic-gradient({color} 0deg {deg:.0f}deg, '
            f'rgba(255,255,255,.08) {deg:.0f}deg 360deg);">'
            f'<div class="ring-inner"><span>{prob*100:.0f}%</span><small>hit chance</small></div>'
            f'</div>'
            f'<div class="verdict"><h2>{title}</h2><p>{line}</p>'
            f'<span class="badge">{confidence}</span></div>'
            f'</div>',
            unsafe_allow_html=True)
    else:
        st.info("Set the movie details in the sidebar, then click **Predict Box-Office Outcome**.")

    st.markdown(
        '<div class="note">Predictions are estimates based on historical data, '
        'not guarantees.</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("""
**How it works**
- A movie is labelled a **HIT** if its gross is at least 1.5x its budget, otherwise a **FLOP**.
- The model is a tuned **Gradient Boosting** classifier (test ROC-AUC ≈ 0.87, accuracy ≈ 78%).
- Inputs: budget, runtime, year, expected IMDb score and votes, release month, genre,
  rating, country, company, director and star.

**Reading the result**
- The ring shows the model's estimated probability that the movie is a hit.
- *Close call* means the probability is near 50%, so treat the prediction with caution.
""")
