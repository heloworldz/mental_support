import streamlit as st
import requests
import os

st.title("Hugging Face Token & API Test")

# Load token from Streamlit secrets or environment
try:
    HF_TOKEN = st.secrets["hf_token"]
except:
    HF_TOKEN = os.getenv("hf_token")

st.write("Token loaded?", bool(HF_TOKEN))

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

if st.button("Test API"):
    payload = {"inputs": "Hello"}
    with st.spinner("Sending request..."):
        response = requests.post(API_URL, headers=headers, json=payload)
    st.write("Status code:", response.status_code)
    st.write("Response:", response.text)
