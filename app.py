import requests
import os

HF_TOKEN = os.getenv("hf_token")  # Make sure this matches your env var name
print("Token loaded?", bool(HF_TOKEN))

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

payload = {"inputs": "Hello"}
response = requests.post(API_URL, headers=headers, json=payload)
print("Status code:", response.status_code)
print("Response:", response.text)
