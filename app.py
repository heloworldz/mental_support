import streamlit as st
import requests
import json
import re
from textblob import TextBlob

# -------------- Config -------------------
HF_TOKEN = st.secrets["HF_TOKEN"]
MODEL_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# System prompt to steer the AI towards compassionate mental health support
SYSTEM_PROMPT = (
    "You are a compassionate mental-health assistant. "
    "Never repeat the user's words verbatim. "
    "If the user talks about self-harm or violent thoughts, respond with empathy "
    "and encourage seeking professional help. "
    "Answer briefly, supportively, and positively.\n\n"
)

# Regex pattern for crisis detection
CRISIS_PATTERN = re.compile(
    r"\b(kill\s+myself|suicid(?:e|al)|end\s+my\s+life|die\s+by\s+suicide|"
    r"don't\s+want\s+to\s+live|take\s+my\s+own\s+life|hurt\s+myself)\b", re.I
)

# ----------- Functions -------------------

def is_crisis(text: str) -> bool:
    return bool(CRISIS_PATTERN.search(text))

def query_model(user_input: str) -> str:
    prompt = SYSTEM_PROMPT + f"User: {user_input}\nAssistant:"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 120,
            "temperature": 0.3,
            "top_p": 0.85,
            "stop": ["User:", "\n\n"]
        }
    }
    response = requests.post(MODEL_API_URL, headers=HEADERS, json=payload, timeout=40)
    response.raise_for_status()
    result = response.json()
    text = result[0].get("generated_text", "")
    reply = text.split("Assistant:")[-1].strip()
    return reply.replace(user_input, "").strip() or "I'm here to listen."

# ----------- Streamlit UI -----------------

st.set_page_config(page_title="AI Mental Health Chatbot", layout="centered")
st.title("🧠 AI Mental Health Chatbot")
st.caption("Supportive, non-clinical conversation. For emergencies, call your local helpline.")

user_message = st.text_input("Type your message here:")

if user_message:
    # Show sentiment
    polarity = TextBlob(user_message).sentiment.polarity
    if polarity > 0.1:
        st.info("😊 Sentiment: Positive")
    elif polarity < -0.1:
        st.info("☹️ Sentiment: Negative")
    else:
        st.info("😐 Sentiment: Neutral")

    if is_crisis(user_message):
        st.warning(
            "⚠️ It sounds like you're in a very difficult place. "
            "You **matter** and you’re **not alone**.\n\n"
            "• **India (AASRA 24×7): 915 298 7821**\n"
            "• Worldwide helplines: https://findahelpline.com\n\n"
            "Please consider talking to a trusted friend or professional."
        )
    else:
        with st.spinner("Generating response..."):
            try:
                reply = query_model(user_message)
                st.markdown("### 🤖 AI Response")
                st.write(reply)
            except Exception as e:
                st.error(f"Error contacting model API: {e}")
