import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration — always first line
st.set_page_config(
    page_title="House Price Estimator",
    page_icon="🏠",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model    = joblib.load('models/house_price_model.pkl')
    features = joblib.load('models/feature_names.pkl')
    return model, features

model, feature_names = load_model()

# ── Header ────────────────────────────────────────────────
st.title("🏠 House Price Estimation System")
st.markdown("##### Predict Ames, Iowa house prices using a tuned Gradient Boosting model")
st.markdown("---")

# ── Sidebar inputs ────────────────────────────────────────
st.sidebar.header("🏗️ Enter House Details")

overall_qual  = st.sidebar.slider(
    "Overall Quality (1=Poor, 10=Excellent)", 1, 10, 6)

gr_liv_area   = st.sidebar.slider(
    "Above-Ground Living Area (sqft)", 400, 5000, 1500)

total_bsmt_sf = st.sidebar.slider(
    "Basement Area (sqft)", 0, 3000, 800)

first_flr_sf  = st.sidebar.slider(
    "1st Floor Area (sqft)", 400, 4000, 1000)

second_flr_sf = st.sidebar.slider(
    "2nd Floor Area (sqft)", 0, 2000, 0)

garage_cars   = st.sidebar.selectbox(
    "Garage Capacity (cars)", [0, 1, 2, 3, 4], index=2)

year_built    = st.sidebar.slider(
    "Year Built", 1872, 2010, 1990)

full_bath     = st.sidebar.selectbox(
    "Full Bathrooms", [1, 2, 3, 4], index=1)

fireplaces    = st.sidebar.selectbox(
    "Fireplaces", [0, 1, 2, 3], index=0)

yr_sold       = st.sidebar.selectbox(
    "Year Sold", [2006, 2007, 2008, 2009, 2010], index=4)

# ── Block 4 — Engineered features (only once!) ────────────
total_sf      = total_bsmt_sf + first_flr_sf + second_flr_sf
total_baths   = full_bath
house_age     = yr_sold - year_built
has_garage    = 1 if garage_cars > 0 else 0
has_fireplace = 1 if fireplaces > 0 else 0
qual_x_area   = overall_qual * total_sf

# ── Block 5 — Build input dataframe (THIS WAS MISSING!) ───
input_dict = {name: 0 for name in feature_names}

input_dict.update({
    'OverallQual':  overall_qual,
    'GrLivArea':    gr_liv_area,
    'TotalBsmtSF':  total_bsmt_sf,
    '1stFlrSF':     first_flr_sf,
    '2ndFlrSF':     second_flr_sf,
    'GarageCars':   garage_cars,
    'YearBuilt':    year_built,
    'FullBath':     full_bath,
    'Fireplaces':   fireplaces,
    'TotalSF':      total_sf,
    'TotalBaths':   total_baths,
    'HouseAge':     house_age,
    'HasGarage':    has_garage,
    'HasFireplace': has_fireplace,
    'QualXArea':    qual_x_area,
})

input_df = pd.DataFrame([input_dict])

# ── Block 6 — Predict and display ─────────────────────────
log_pred = model.predict(input_df)[0]
price    = np.expm1(log_pred)

col1, col2, col3 = st.columns(3)
col1.metric("💰 Estimated Price",  f"${price:,.0f}")
col2.metric("📐 Total Sq Footage", f"{total_sf:,} sqft")
col3.metric("🏚️ House Age",        f"{house_age} years")

low  = price * 0.87
high = price * 1.13

st.success(
    f"Estimated price: **${price:,.0f}**  |  "
    f"Likely range: ${low:,.0f} – ${high:,.0f}"
)

with st.expander("📋 View Full Input Summary"):
    summary = pd.DataFrame({
        'Feature': ['Quality', 'Living Area', 'Total SF',
                    'Garage Cars', 'House Age', 'Fireplaces'],
        'Value':   [overall_qual, gr_liv_area, total_sf,
                    garage_cars, house_age, fireplaces]
    })
    st.dataframe(summary, use_container_width=True)