import streamlit as st
import pandas as pd
from src.predict import predict_rating

# Load your dataset

# Corrected line with encoding
data = pd.read_csv(r'C:\Users\DELL\Downloads\movies.CSV', encoding='latin1')
# You can now use `data` if needed later

st.title("🎬 Movie Rating Prediction App")

# Take user input
duration = st.number_input("Enter movie duration (in minutes):", min_value=1)
votes = st.number_input("Enter number of votes:", min_value=1)

# Predict button
if st.button("Predict Rating"):
    prediction = predict_rating(duration, votes)
    st.success(f"Predicted Movie Rating: {prediction:.2f}")
