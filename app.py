import requests
import os
import streamlit as st

HF_TOKEN = os.getenv("hf_token")
API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-13b-chat-hf"

st.write(f"Token loaded? {'Yes' if HF_TOKEN else 'No'}")

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_hf_api(prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            return response.json()[0]['generated_text']
        except Exception:
            return "Sorry, I could not generate a response."
    else:
        return f"API error: {response.status_code}"

response = "Hello, world!"  # default message before user input

user_input = st.text_input("Your message:")

if user_input:
    response = query_hf_api(user_input)

st.write(response)
