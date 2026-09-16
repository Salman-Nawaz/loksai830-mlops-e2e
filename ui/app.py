import streamlit as st
import requests

# ⚠️ When running locally
API_URL = "http://salman-api-svc:8000/predict"

st.title("Loksai MLOps Feedback Analyzer")

text = st.text_area("Enter feedback")

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter some text!")

    else: 
        response = requests.post(API_URL, params={"text": text})

        if response.status_code == 200:
            result = response.json()

            sentiment = result["sentiment"]
        

            st.markdown(
                f"""
                ### 🧠 Prediction Result
                **Sentiment:** `{sentiment.capitalize()}`
                """
            )
        else:
            st.error("Failed to get prediction from API")
