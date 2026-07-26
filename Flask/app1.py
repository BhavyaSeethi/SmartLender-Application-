import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('rdf.pkl', 'rb'))
scale = pickle.load(open('scale1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('input.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Read inputs from form by name for reliability
    features = [
        float(request.form.get('Gender', 0)),
        float(request.form.get('Married', 0)),
        float(request.form.get('Dependents', 0)),
        float(request.form.get('Education', 0)),
        float(request.form.get('Self_Employed', 0)),
        float(request.form.get('ApplicantIncome', 0)),
        float(request.form.get('CoapplicantIncome', 0)),
        float(request.form.get('LoanAmount', 0)),
        float(request.form.get('Loan_Amount_Term', 360)),
        float(request.form.get('Credit_History', 1)),
        float(request.form.get('Property_Area', 0)),
    ]

    names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
             'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
             'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    data = pd.DataFrame([features], columns=names)

    # Scale and predict
    data_scaled = scale.transform(data)
    prediction = int(model.predict(data_scaled)[0])

    if prediction == 0:
        result = "Loan will NOT be Approved ❌"
    else:
        result = "Loan will be Approved ✅"

    return render_template('submit.html', result=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
