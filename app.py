from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scale.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict")
def predict():
    return render_template("predict.html")

@app.route("/submit", methods=["POST"])
def submit():
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    scaled_features = scaler.transform(final_features)
    prediction = model.predict(scaled_features)[0]

    if prediction == 1:
        result = "Loan will be Approved ✅"
    else:
        result = "Loan will NOT be Approved ❌"

    return render_template("submit.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
