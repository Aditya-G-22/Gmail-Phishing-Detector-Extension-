import joblib
import pandas as pd

MODEL_PATH = "phishing_rf_pipeline.joblib"

_pipeline = None

def get_pipeline() :
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline

#----------FEATURE COLUMN-----------

FEATURE_COLUMNS = [
    "length_url",
    "length_hostname",
    "nb_dots",
    "nb_hyphens",
    "nb_at",
    "nb_qm",
    "nb_and",
    "nb_or",
    "domain_registration_length",
    "domain_age_value",
    "web_traffic",
    "page_rank",
    "ip",
    "whois_registered_domain",
    "dns_record",
    "google_index",
    "domain_in_title",
    "domain_with_copyright",
    "domain_age_known"
]


def predict_phishing(features: dict, threshold: float = 0.35) -> dict:
    """
    Predict whether a URL is phishing using a trained Random Forest pipeline.
    """

    #----------VALIDATE INPUT-----------

    missing = set(FEATURE_COLUMNS) - set(features.keys())
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    #----------CONVERT INPUT TO DATAFRAME-------------

    X = pd.DataFrame([features], columns = FEATURE_COLUMNS)

    #----------PREDICT PROBABILITY---------------

    pipeline = get_pipeline()
    phishing_prob = pipeline.predict_proba(X)[0][1]

    #-----------APPLY THRESHOLD----------------

    prediction = int(phishing_prob >= threshold)

    #-----------RETURN STRUCTURED RESULT-----------

    return {
        "phishing_probability": float(phishing_prob),
        "is_phishing": prediction,
        "threshold": threshold
    }
