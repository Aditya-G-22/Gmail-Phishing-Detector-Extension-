import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

with open("email_temperature.json", "r") as f:
    TEMPERATURE = json.load(f)["temperature"]


MODEL_PATH = "email_phishing_model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

def tokenize_text(text) :
    return tokenizer (
        text,
        padding = True,
        truncation = True,
        max_length = 256,
        return_tensors = "pt" 
    )

def predict_email(text):
    inputs = tokenize_text(text)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        logits = logits / TEMPERATURE
        probs = torch.softmax(logits, dim=1)

    phishing_prob = probs[0][1].item()

    label = "phishing" if phishing_prob >= 0.5 else "legitimate"

    return {
        "label" : label,
        "phishing_probability" : phishing_prob
    }