import os
os.environ["XDG_CONFIG_HOME"] = "/tmp"

import streamlit as st
from transformers import pipeline
from textblob import TextBlob
import numpy as np
import speech_recognition as sr
import tempfile

# Hugging Face token (securely stored in Streamlit secrets)
hf_token = st.secrets["hf_token"]

st.set_page_config(page_title="Mental Health Chatbot", layout="centered")

# --- Load model ---
@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="distilgpt2",
        tokenizer="distilgpt2",
        use_auth_token=hf_token
    )

generator = load_model()

# --- Crisis/Sensitive message check ---
def is_sensitive_message(msg):
    crisis_keywords = [
        "kill myself", "end my life", "suicide",
        "i want to die", "don’t want to live", "i hate my life"
    ]
    return any(kw in msg.lower() for kw in crisis_keywords)

# --- Generate safe AI response ---
def generate_safe_response(user_input):
    safe_prompt = f"""You are a supportive mental health assistant.
Respond with empathy and care.
Do NOT repeat the user's message. Instead, offer kind, encouraging words.

User: {user_input}
Assistant:"""

    raw_output = generator(
        safe_prompt,
        max_length=150,
        do_sample=True,
        temperature=0.8,
        pad_token_id=50256  # for distilgpt2
    )[0]['generated_text']

    # Remove user input from AI response if echoed
    reply = raw_output.split("Assistant:")[-1].strip()
    return reply.replace(user_input, "").strip()

# --- UI Elements ---
st.title("🧠 Mental Health Chatbot")

# Initialize session state
for key in ["last_input", "show_sentiment", "show_affirmations", "show_meditation"]:
    if key not in st.session_state:
        st.session_state[key] = False if key.startswith("show_") else ""

# Sidebar toggles
st.sidebar.title("🛠️ Tools")
st.session_state.show_sentiment = st.sidebar.checkbox("Show Sentiment", value=st.session_state.show_sentiment)
st.session_state.show_affirmations = st.sidebar.checkbox("Show Positive Affirmations", value=st.session_state.show_affirmations)
st.session_state.show_meditation = st.sidebar.checkbox("Show Guided Meditation", value=st.session_state.show_meditation)

# --- Voice Input ---
st.subheader("🎤 Voice Input")
audio_file = st.file_uploader("Upload a WAV audio file", type=["wav"])
text = ""

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    recognizer = sr.Recognizer()
    with sr.AudioFile(tmp_path) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        st.success(f"Recognized Text: {text}")
    except sr.UnknownValueError:
        st.error("Could not understand the audio")
    except sr.RequestError:
        st.error("Speech recognition service failed")

# --- Text Input ---
user_input = st.text_input("🗣️ Or type your message:", text)

# Reset options if input changes
if user_input and user_input != st.session_state.last_input:
    st.session_state.show_sentiment = False
    st.session_state.show_affirmations = False
    st.session_state.show_meditation = False
    st.session_state.last_input = user_input

# --- Sentiment Analysis ---
if st.session_state.show_sentiment and user_input:
    sentiment = TextBlob(user_input).sentiment.polarity
    if sentiment > 0:
        st.info("😊 Sentiment: Positive")
    elif sentiment < 0:
        st.info("☹️ Sentiment: Negative")
    else:
        st.info("😐 Sentiment: Neutral")

# --- Chatbot Response ---
if user_input:
    with st.spinner("Thinking..."):
        if is_sensitive_message(user_input):
            st.markdown("### ⚠️ Crisis Alert")
            st.warning("🚨 It sounds like you're in crisis. Please seek immediate help:\n\n📞 **Suicide Prevention Helpline India**: 9152987821\n🌐 [Find help near you](https://findahelpline.com)")
        else:
            try:
                response = generate_safe_response(user_input)
                st.markdown("### 🤖 AI Response")
                st.write(response)
            except Exception as e:
                st.error("Something went wrong while generating the response.")

# --- Affirmations ---
if st.session_state.show_affirmations:
    st.markdown("### 🌟 Positive Affirmations")
    affirmations = [
        "You are capable and strong.",
        "Your feelings are valid.",
        "You are not alone.",
        "You can handle anything that comes your way.",
        "This too shall pass."
    ]
    st.write(np.random.choice(affirmations))

# --- Guided Meditation ---
if st.session_state.show_meditation:
    st.markdown("### 🧘 Guided Meditation")
    st.markdown("""
    Take a deep breath in... and out.  
    Let go of tension.  
    Feel the air fill your lungs.  
    You are safe, calm, and in control.  
    Let your thoughts drift like clouds.  
    Return to the present moment gently.
    """)
