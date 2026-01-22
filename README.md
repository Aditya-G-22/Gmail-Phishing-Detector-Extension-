📧 Gmail Phishing Detector Extension

A real-time Gmail phishing detection system that analyzes email content and embedded URLs using machine learning models, and displays inline security warnings directly inside Gmail via a Chrome extension.

This project combines NLP (Transformers), URL-based ML detection, FastAPI, and DOM-level Gmail integration to reduce phishing risks for end users.

🚀 Key Features

🔍 Real-time Gmail scanning using DOM parsing + MutationObserver

🧠 Email phishing detection using a fine-tuned DistilBERT model

🔗 URL phishing detection using a trained Random Forest pipeline

🧩 Model fusion logic to combine email + URL risk intelligently

⚠️ Inline Gmail warning banner (High Risk / Suspicious)

📉 Reduced false positives with probability fusion

🌐 FastAPI backend for inference

🧪 Tested on real emails (Quora, Discord, Spotify, Google)

🧠 Machine Learning Models
1️⃣ Email Phishing Model

Model: DistilBERT

Task: Binary classification (phishing / legitimate)

Input: Raw email text

Output: Phishing probability

Framework: transformers, torch

2️⃣ URL Phishing Model

Model: Random Forest

Features:

URL length

Hostname length

Special characters

IP presence

Domain heuristics

Output: URL phishing probability
