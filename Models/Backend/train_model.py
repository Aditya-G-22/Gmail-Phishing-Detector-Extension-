#Imports
#Configuration
#Load Data
#Encode Labels
#Create X and y
#Train-Test Split
#Preprocessing
#Model
#Build pipeline
#Train model
#Evaluate model

#---------IMPORTS-------------
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#---------CONFIGURATION--------

data_path = "emails.csv"
Test_Size = 0.2
Random_State = 42
threshold = 0.35

binary_features = [
    "ip",
    "whois_registered_domain",
    "dns_record",
    "google_index",
    "domain_in_title",
    "domain_with_copyright",
    "domain_age_known"
]
continuous_features = [
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
    "page_rank"
]

feature_columns = continuous_features + binary_features

#------------LOAD DATA-----------------

df = pd.read_csv(data_path)

#------------ENCODE LABELS-----------------

df["status"] = df["status"].map({
    "legitimate": 0,
    "phishing": 1
})

if df["status"].isnull().any():
    raise ValueError("Label encoding failed")

#-----------FEATURE ENGINEERING----------

df["domain_age_known"] = (df["domain_age"] != -1).astype(int)

df["domain_age_value"] = df["domain_age"].where(
    df["domain_age"] != -1, 0
)


#------------CREATE X AND Y-----------------

X = df[feature_columns]
y = df["status"]

#------------TRAIN-TEST SPLIT-----------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = Test_Size,
    random_state = Random_State,
    stratify=y
)

#------------PREPROCESSING-----------------

preprocessor = ColumnTransformer (
    transformers = [
        ("num", StandardScaler(), continuous_features),
        ("bin", "passthrough", binary_features)
    ]
)

#------------MODEL-----------------

model = RandomForestClassifier (
    n_estimators = 200,
    max_depth = 15,
    min_samples_leaf = 10,
    class_weight = "balanced",
    random_state = Random_State,
    n_jobs = 1
)

#------------BUILD PIPELINE-----------------

pipeline = Pipeline(steps = [
    ("preprocessing", preprocessor),
    ("classifier", model)
])

#---------TRAIN THE MODEL-------------

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "phishing_rf_pipeline.joblib")

#---------EVALUATE THE MODEL------------

y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred_custom = (y_prob >= threshold).astype(int)
print(classification_report(y_test, y_pred_custom))

cm_custom =  confusion_matrix(y_test, y_pred_custom)
print("Confusion Matrix (threshold-tuned):")
print(cm_custom)


