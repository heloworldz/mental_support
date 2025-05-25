import os
import requests
import streamlit as st

# Get your HF API token from environment variable
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face API token is not set in environment variable HF_TOKEN.")
    st.stop()

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_hf_api(messages):
    """
    Query the HF Inference API with chat messages formatted as list of dicts.
    """
    payload = {"inputs": messages}

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            # The response is usually a list with 'generated_text' field
            # For chat models, the API returns [{'generated_text': "..."}]
            data = response.json()
            return data[0]['generated_text']
        except Exception as e:
            return f"Error parsing response: {e}"
    else:
        return f"API error {response.status_code}: {response.text}"

st.title("Hugging Face Llama 2 Chatbot")

user_input = st.text_input("Your message:")

if user_input:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]

    response = query_hf_api(messages)
    st.write(response)
