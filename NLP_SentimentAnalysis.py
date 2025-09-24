import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from streamlit_option_menu import option_menu
from scipy.sparse import hstack

import pickle
from sklearn import preprocessing
from xgboost import XGBClassifier
BASE_DIR = os.path.dirname(__file__)

#Initialize LableEncoder
with open(os.path.join(BASE_DIR,'labelencoder.pkl'), 'rb') as file:
    encoders = pickle.load(file)

# Load vectorizer
with open(os.path.join(BASE_DIR, 'tfidf_vectorizer_model.pkl'), 'rb') as filename:
    tfidf_model = pickle.load(filename)

# Load the model from file
with open(os.path.join(BASE_DIR, 'randfor_model.pkl'), 'rb') as filename:
    loaded_model = pickle.load(filename)

df = pd.read_csv(os.path.join(BASE_DIR,"chatgpt_style_reviews_dataset.xlsx - Sheet1.csv"))

# Styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 34px;
        font-weight: bold;
        color: #4CAF50;
        text-shadow: 2px 2px 5px rgba(76, 175, 80, 0.4);
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        #color: #ddd;
        margin-bottom: 20px;
    }
    .stButton button {
        background: linear-gradient(to right, #4CAF50, #388E3C);
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton button:hover {
        background: linear-gradient(to right, #388E3C, #2E7D32);
    }
    .result-card {
        background: rgba(0, 150, 136, 0.1);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0px 2px 8px rgba(0, 150, 136, 0.2);
    }
    .success-banner {
        background: linear-gradient(to right, #2E7D32, #1B5E20);
        color: white;
        padding: 15px;
        font-size: 18px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        margin-top: 15px;
        box-shadow: 0px 2px 8px rgba(0, 150, 136, 0.5);
    }
    </style>
""", unsafe_allow_html=True)


with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Home", "Projects", "About"],  # required
        icons=["house", "book", "person"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
    )
if selected == "Home":
    st.markdown('<h1 class="main-title">🌐 AI Echo: Your Smartest Conversational Partner</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title"> It aims to analyze user reviews of a ChatGPT application and classify them as positive, neutral, or negative based on the sentiment expressed.</p>', unsafe_allow_html=True)

if selected == 'Projects':
    st.markdown(
    """
    <h3 style='text-align: center; color: #0B4242; text-shadow: 2px 2px 5px gray; word-spacing: 5px; border: 2px solid #333;
    border-radius: 8px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3); background-color: #F5F5F5;
    padding: 10px; margin: 15px;'>
        Please provide your feedback
    </h3>
    """,
    unsafe_allow_html=True)
    # Features
    features = ['title','review','rating', 'helpful_votes', 'review_length', 'platform', 'language',
       'location', 'version', 'verified_purchase' ]
    if "helpful_votes" not in st.session_state:
        st.session_state.helpful_votes = 0
    title=st.text_input("Title")
    review = st.text_input("Review")
    if st.button("👍"):
        st.session_state.helpful_votes += 1
        st.write("Helpful Votes:", st.session_state.helpful_votes)
    rating=st.slider("Ratings",min_value= 1, max_value=5)
    platform = st.selectbox("Platform", sorted(df["platform"].dropna().unique()))
    language = st.selectbox("Language", sorted(df["language"].dropna().unique()))
    verified_purchase = st.selectbox("Verified Purchase", sorted(df["verified_purchase"].dropna().unique()))
    location = st.text_input("Location")
    version = st.text_input("Version")
    if st.button("Submit Review"):
        full_text = title + " " + review
        text_features = tfidf_model.transform([full_text])
        
        structured = pd.DataFrame([{
        "rating": rating,
        "helpful_votes": st.session_state.helpful_votes,
        "review_length": len(review.split()),
        "platform": platform,
        "language": language,
        "location": location,
        "version": version,
        "verified_purchase": verified_purchase}])
        for col, encoder in encoders.items():
            if col in structured.columns:
                structured[col] = encoder.transform(structured[col])
        structured = structured[structured.columns]
        final_features = hstack([text_features, structured])

        pred = loaded_model.predict(final_features)
        if pred[0]==0:
            st.warning("OOPs we got negative review")
        elif pred[0]==1:
            st.success("we got neutral review, lot to improve")
        elif pred[0]==2:
            st.balloons()
            st.success("success we got POSITIVE review")














