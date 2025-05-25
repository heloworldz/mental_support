import streamlit as st
import requests
import re
from textblob import TextBlob

MODEL_API_URL = "https://api-inference.huggingface.co/models/emozilla/mental-health-ai"

# No token needed for free models on Hugging Face (if public)
HEADERS = {}

# Regex for basic crisis detection
CRISIS_PATTERN = re.compile(
    r"\b(kill\s+myself|suicid(?:e|al)|end\s+my\s+life|don't\s+want\s+to\s+live|"
    r"take\s+my\s+own\s+life|hurt\s+myself|die)\b", re.I
)

def is_crisis(text: str) -> bool:
    return bool(CRISIS_PATTERN.search(text))

def query_model(user_input: str) -> str:
    prompt = f"User: {user_input}\nTherapist:"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.6,
            "top_p": 0.9,
            "stop": ["User:", "\n"]
        }
    }
    response = requests.post(MODEL_API_URL, headers=HEADERS, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    return result[0]["generated_text"].split("Therapist:")[-1].strip()

# --- Streamlit UI ---
st.set_page_config(page_title="🧠 AI Mental Health Chatbot", layout="centered")
st.title("🧠 AI Mental Health Chatbot")
st.caption("Free, empathetic mental health support (not a replacement for professional help)")

user_message = st.text_input("Type your message here:")

if user_message:
    # Sentiment
    polarity = TextBlob(user_message).sentiment.polarity
    if polarity > 0.1:
        st.info("😊 Sentiment: Positive")
    elif polarity < -0.1:
        st.info("☹️ Sentiment: Negative")
    else:
        st.info("😐 Sentiment: Neutral")

    # Check for crisis
    if is_crisis(user_message):
        st.warning(
            "⚠️ It sounds like you're in a very difficult place. "
            "You're **not alone** and help is available.\n\n"
            "• India (AASRA 24×7): 9152987821\n"
            "• Worldwide helplines: https://findahelpline.com"
        )
        st.stop()

    with st.spinner("Thinking..."):
        try:
            response = query_model(user_message)
            st.markdown("### 🤖 AI Response")
            st.write(response)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# Optional: Add self-care tips
if st.button("🩺 Show Self-care Tips"):
    st.markdown("""
    ### 🩺 Self-Care Tips
    - 🧘 Practice deep breathing or meditation
    - 🏃 Go for a short walk
    - 📖 Write down your thoughts
    - 🗣️ Talk to someone you trust
    - 💧 Drink water and eat well
    """)
