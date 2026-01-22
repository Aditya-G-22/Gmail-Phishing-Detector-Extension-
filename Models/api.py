from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from email_inference import predict_email
from url_inference import predict_phishing

import re

#------------------APP--------------------------

app = FastAPI(title="Email Phishing Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#-------------------Helpers----------------------

URL_REGEX = r"https?://[^\s]+"

def extract_urls(text: str) -> list[str]:
    return re.findall(URL_REGEX, text)

def is_ip(domain: str) -> int:
    parts = domain.split(".")
    return int(len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts))


def url_to_features(url: str) -> dict:
    domain = url.replace("https://", "").replace("http://", "").split("/")[0]
    return {
        "length_url": len(url),
        "length_hostname": len(url.split("/")[2]),
        "nb_dots": url.count("."),
        "nb_hyphens": url.count("-"),
        "nb_at": url.count("@"),
        "nb_qm": url.count("?"),
        "nb_and": url.count("&"),
        "nb_or": url.count("|"),
        "domain_registration_length": 0,  # 
        "domain_age_value": 0,
        "web_traffic": 0,
        "page_rank": 0,
        "ip": is_ip(domain),
        "whois_registered_domain": 1,
        "dns_record": 1,
        "google_index": 1,
        "domain_in_title": 0,
        "domain_with_copyright": 0,
        "domain_age_known": 0
    }

#----------------------MODEL Fusion------------------------

def fuse_predictions(email_prob: float, url_probs: list[float]):
    if not url_probs:
        return email_prob, "email_only"

    max_url_prob = max(url_probs)

    # Strong malicious URL dominates
    if max_url_prob >= 0.85:
        return max_url_prob, "malicious_url"

    # Weighted fusion
    fused_prob = (0.6 * email_prob) + (0.4 * max_url_prob)
    return fused_prob, "combined"

def risk_label(prob: float) -> str:
    if prob >= 0.8:
        return "high_risk"
    elif prob >= 0.4:
        return "suspicious"
    else:
        return "legitimate"
    
#------------------------SCHEMAS------------------------------

class EmailRequest(BaseModel) :
    text : str

#-------------------Routes----------------------------

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict/email", tags=["Email Detection"])
def predict_email_route(request: EmailRequest):
                # ----------------Email model------------------------------
    email_result = predict_email(request.text)
    email_prob = email_result["phishing_probability"]

                #-----------------Extract URLs----------------------------
    urls = extract_urls(request.text)

                #-----------------URL model----------------------------
    url_probs = []
    for url in urls:
        features = url_to_features(url)
        url_result = predict_phishing(features)
        url_probs.append(url_result["phishing_probability"])

                #-----------------Fuse---------------------------
    final_prob, source = fuse_predictions(email_prob, url_probs)

                #-----------------Label-------------------------
    label = risk_label(final_prob)

    return {
        "label": label,
        "final_probability": round(final_prob, 4),
        "email_probability": round(email_prob, 4),
        "url_probability": round(max(url_probs), 4) if url_probs else None,
        "decision_source": source,
        "urls_detected": urls
    }
