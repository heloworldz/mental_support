###############################################################################
#  mental_health_chatbot.py  –  SAFE VERSION  (Streamlit 1.34+)
###############################################################################
import os, re, json, tempfile, requests
os.environ["XDG_CONFIG_HOME"] = "/tmp"

import streamlit as st
from textblob import TextBlob
import numpy as np
import speech_recognition as sr

# ─────────────────────────────  CONFIG  ───────────────────────────────────────
HF_TOKEN   = st.secrets["HF_TOKEN"]          # add in Settings ▸ Secrets
MODEL_URL  = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

st.set_page_config(page_title="Mental Health Chatbot", layout="centered")
st.title("🧠 Mental Health Chatbot")
st.caption("This tool offers supportive, non-clinical conversation. "
           "For emergencies call your local helpline.")

# ─────────────────────────  SESSION STATE  ────────────────────────────────────
for k in ("last_input", "show_sentiment", "show_affirmations", "show_meditation"):
    st.session_state.setdefault(k, False if k.startswith("show_") else "")

# ───────────────────────────  SIDEBAR  ────────────────────────────────────────
st.sidebar.title("🛠️ Tools")
st.session_state.show_sentiment     = st.sidebar.checkbox("Show sentiment",         st.session_state.show_sentiment)
st.session_state.show_affirmations  = st.sidebar.checkbox("Positive affirmations",  st.session_state.show_affirmations)
st.session_state.show_meditation    = st.sidebar.checkbox("Guided meditation",      st.session_state.show_meditation)

# ───────────────────────  CRISIS DETECTION  ───────────────────────────────────
CRISIS_PAT  = re.compile(r"\b("
    r"kill\s+myself|suicid(?:e|al)|end\s+my\s+life|die\s+by\s+suicide|"
    r"don'?t\s+want\s+to\s+live|take\s+my\s+own\s+life|hurt\s+myself"
")\b", re.I)

def in_crisis(msg:str)->bool:
    return bool(CRISIS_PAT.search(msg))

# ──────────────────────  HUGGING FACE GENERATION  ─────────────────────────────
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

SYSTEM_PROMPT = (
    "You are a compassionate mental-health assistant. "
    "NEVER repeat the user's words verbatim. "
    "If the user asks self-harm or violent questions, respond with empathy "
    "and encourage seeking professional help. Otherwise, answer briefly, "
    "supportively, and positively.\n\n"
)

def generate_mistral(user_msg:str)->str:
    payload = {
        "inputs":  SYSTEM_PROMPT + f"User: {user_msg}\nAssistant:",
        "parameters": {
            "max_new_tokens": 160,
            "temperature": 0.7,
            "top_p": 0.95,
            "stop": ["User:", "\n\n"]
        }
    }
    r = requests.post(MODEL_URL, headers=HEADERS, data=json.dumps(payload), timeout=40)
    r.raise_for_status()
    text = r.json()[0]["generated_text"]
    # Grab everything after the last "Assistant:" (robust to reuse of system prompt)
    reply = text.split("Assistant:")[-1].strip()
    # Strip any accidental echo
    return reply.replace(user_msg, "").strip()

# ───────────────────────────  VOICE INPUT  ────────────────────────────────────
st.subheader("🎤 Voice Input (WAV)")
audio_file = st.file_uploader("Upload a WAV file", type=["wav"], label_visibility="collapsed")
initial_text = ""

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        wav_path = tmp.name
    r = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)
    try:
        initial_text = r.recognize_google(audio)
        st.success(f"Recognised: “{initial_text}”")
    except sr.UnknownValueError:
        st.error("Could not understand the audio.")
    except sr.RequestError:
        st.error("Speech-recognition service failed.")

# ─────────────────────────────  TEXT INPUT  ───────────────────────────────────
user_input = st.text_input("🗣️ Or type your message:", value=initial_text)

# Auto-reset feature toggles when the text changes
if user_input and user_input != st.session_state.last_input:
    for k in ("show_sentiment", "show_affirmations", "show_meditation"):
        st.session_state[k] = False
    st.session_state.last_input = user_input

# ─────────────────────────  SENTIMENT  ────────────────────────────────────────
if st.session_state.show_sentiment and user_input:
    pol = TextBlob(user_input).sentiment.polarity
    if   pol >  0.1: st.info("😊 Sentiment: Positive")
    elif pol < -0.1: st.info("☹️ Sentiment: Negative")
    else:            st.info("😐 Sentiment: Neutral")

# ───────────────────────  MAIN RESPONSE FLOW  ────────────────────────────────
if user_input:
    with st.spinner("Thinking…"):
        if in_crisis(user_input):
            st.markdown("### ⚠️ Crisis Alert")
            st.warning(
                "It sounds like you're in a very difficult place. "
                "You **matter** and you’re **not alone**.\n\n"
                "• **India (AASRA 24×7): 915 298 7821**\n"
                "• Worldwide helplines: <https://findahelpline.com>\n\n"
                "Consider talking to a trusted friend or a professional right now."
            )
        else:
            try:
                bot_reply = generate_mistral(user_input)
                st.markdown("### 🤖 AI Response")
                st.write(bot_reply if bot_reply else "I'm here to listen.")
            except Exception as e:
                st.error(f"Generation failed: {e}")

# ──────────────────  OPTIONAL EXTRAS (AFFIRM & MEDITATE)  ─────────────────────
if st.session_state.show_affirmations:
    st.markdown("### 🌟 Positive Affirmations")
    st.write(np.random.choice([
        "You are capable and strong.",
        "Your feelings are valid.",
        "You are not alone.",
        "You can handle anything that comes your way.",
        "This too shall pass.",
        "You deserve kindness, especially from yourself."
    ]))

if st.session_state.show_meditation:
    st.markdown("### 🧘 Guided Meditation")
    st.markdown(
        "Sit comfortably, close your eyes, and take a slow breath in … and out …\n\n"
        "Notice the chair beneath you. Let your shoulders drop. Let thoughts float "
        "by like clouds. You are safe, and this moment belongs to you."
    )
###############################################################################
