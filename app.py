import streamlit as st
from transformers import pipeline
import base64
from textblob import TextBlob
import numpy as np
import scipy.io.wavfile
import tempfile

# Hugging Face chatbot pipeline
generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", token=st.secrets["HUGGINGFACE_TOKEN"])

# Analyze sentiment
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positive", polarity
    elif polarity < -0.1:
        return "negative", polarity
    else:
        return "neutral", polarity

# Detect crisis language
def detect_risk(message):
    suicide_phrases = ["kill myself", "want to die", "can't go on", "suicide", "end my life"]
    violence_phrases = ["kill someone", "hurt others", "murder", "attack"]
    lower = message.lower()
    if any(p in lower for p in suicide_phrases):
        return "suicide"
    if any(p in lower for p in violence_phrases):
        return "violence"
    return None

helplines = {
    "india": [
        "📞 iCall: +91 9152987821",
        "📞 AASRA: +91 9820466726",
        "📞 24x7 Helpline: 1800-599-0019"
    ]
}

# Generate chatbot response
def generate_response(user_input):
    risk = detect_risk(user_input)
    if risk == "suicide":
        return ("I'm really sorry you're feeling this way. You're not alone. Please reach out for help.\n" +
                "\n".join(helplines["india"]))

    st.session_state.conversation.append({"role": "user", "content": user_input})
    prompt = "You are a kind and empathetic mental health assistant.\n" + "\n".join([f"User: {m['content']}" if m['role'] == 'user' else f"AI: {m['content']}" for m in st.session_state.conversation]) + "\nAI:"
    response = generator(prompt, max_new_tokens=150, do_sample=True)[0]['generated_text'].split("AI:")[-1].strip()
    st.session_state.conversation.append({"role": "assistant", "content": response})
    return response

# Generate affirmation/meditation
def generate_affirmation():
    response = generator("Give a positive affirmation for someone feeling stressed.", max_new_tokens=50)[0]['generated_text']
    return response.strip()

def generate_meditation_guide():
    response = generator("Give a short guided meditation for relaxation.", max_new_tokens=100)[0]['generated_text']
    return response.strip()

# App Setup
st.set_page_config(page_title="Mental Health Chatbot")
st.title("🧠 Mental Health Support Chatbot")

# Initialize conversation state at the top before any usage
if 'conversation' not in st.session_state:
    st.session_state.conversation = [{"role": "system", "content": "You are a kind and empathetic mental health assistant."}]

# Show chat history
for msg in st.session_state.conversation:
    if msg['role'] in ['user', 'assistant']:
        name = "You" if msg['role'] == "user" else "AI"
        st.markdown(f"**{name}:** {msg['content']}")

# Text input
col1, col2 = st.columns([3, 1])
with col1:
    user_message = st.text_input("How can I help you today?")
with col2:
    upload_audio = st.file_uploader("🎤 Upload voice (.wav)", type=["wav"])

# Process voice input
if upload_audio:
    from speech_recognition import Recognizer, AudioFile
    recognizer = Recognizer()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(upload_audio.read())
        tmpfile_path = tmpfile.name
    try:
        with AudioFile(tmpfile_path) as source:
            audio = recognizer.record(source)
            voice_text = recognizer.recognize_google(audio)
            st.markdown(f"**You (voice):** {voice_text}")
            mood, score = analyze_sentiment(voice_text)
            st.markdown(f"**Detected mood:** {mood} ({score:.2f})")
            with st.spinner("AI is responding..."):
                response = generate_response(voice_text)
                st.markdown(f"**AI:** {response}")
    except Exception as e:
        st.error("Voice recognition failed: " + str(e))

elif user_message:
    mood, score = analyze_sentiment(user_message)
    st.markdown(f"**Detected mood:** {mood} ({score:.2f})")
    with st.spinner("AI is responding..."):
        response = generate_response(user_message)
        st.markdown(f"**AI:** {response}")

# Extra features
col3, col4 = st.columns(2)
with col3:
    if st.button("🌈 Positive Affirmation"):
        st.markdown("**Affirmation:** " + generate_affirmation())
with col4:
    if st.button("🧘 Guided Meditation"):
        st.markdown("**Meditation:** " + generate_meditation_guide())
