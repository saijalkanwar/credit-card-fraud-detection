import joblib

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")
threshold = joblib.load("models/threshold.pkl")

def predict(df):
    X = df.drop(columns=["Class", "Time"], errors="ignore").copy()
    X["Amount"] = scaler.transform(X[["Amount"]])
    probs = model.predict_proba(X)[:, 1]
    return probs, (probs >= threshold).astype(int)