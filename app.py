import os
os.environ["XDG_CONFIG_HOME"] = "/tmp"

import streamlit as st
from transformers import pipeline
from textblob import TextBlob
import numpy as np
import speech_recognition as sr
import tempfile

st.set_page_config(page_title="Mental Health Chatbot", layout="centered")

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilgpt2")

generator = load_model()

st.title("🧠 Mental Health Chatbot")

# Initialize session state variables once
for key in ["last_input", "show_sentiment", "show_affirmations", "show_meditation"]:
    if key not in st.session_state:
        st.session_state[key] = False if key.startswith("show_") else ""

# Sidebar
st.sidebar.title("🛠️ Tools")
show_sentiment = st.sidebar.checkbox("Show Sentiment", value=st.session_state.show_sentiment, key="show_sentiment")
show_affirmations = st.sidebar.checkbox("Show Positive Affirmations", value=st.session_state.show_affirmations, key="show_affirmations")
show_meditation = st.sidebar.checkbox("Show Guided Meditation", value=st.session_state.show_meditation, key="show_meditation")

# Voice input
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

# Text input
user_input = st.text_input("🗣️ Or type your message:", text)

# Auto-reset feature
if user_input and user_input != st.session_state.last_input:
    # New input detected → reset all feature checkboxes
    st.session_state.show_sentiment = False
    st.session_state.show_affirmations = False
    st.session_state.show_meditation = False
    st.session_state.last_input = user_input  # update stored input

# Sentiment
if st.session_state.show_sentiment and user_input:
    sentiment = TextBlob(user_input).sentiment.polarity
    if sentiment > 0:
        st.info("😊 Sentiment: Positive")
    elif sentiment < 0:
        st.info("☹️ Sentiment: Negative")
    else:
        st.info("😐 Sentiment: Neutral")

# Chatbot response
if user_input:
    with st.spinner("Thinking..."):
        response = generator(user_input, max_length=150, do_sample=True, temperature=0.9)[0]['generated_text']
        st.markdown("### 🤖 AI Response")
        st.write(response)

# Affirmations
if st.session_state.show_affirmations:
    st.markdown("### 🌟 Positive Affirmations")
    affirmations = [
        "You are capable and strong.",
        "Your feelings are valid.",
        "You are not alone.",
        "You can handle anything that comes your way."
    ]
    st.write(np.random.choice(affirmations))

# Guided Meditation
if st.session_state.show_meditation:
    st.markdown("### 🧘 Guided Meditation")
    st.markdown("Take a deep breath in... and out. Focus on the present moment. You are safe and in control.")
