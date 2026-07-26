import numpy as np
import pickle
import pandas as pd
import os
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open('rdf.pkl', 'rb'))
scaler = pickle.load(open('scale1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input')
def input_form():
    return render_template('input.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Collect inputs with names
    gender = request.form['Gender']
    married = request.form['Married']
    dependents = int(request.form['Dependents'])
    education = request.form['Education']
    self_employed = request.form['Self_Employed']
    applicant_income = float(request.form['ApplicantIncome'])
    coapplicant_income = float(request.form['CoapplicantIncome'])
    loan_amount = float(request.form['LoanAmount'])
    loan_term = float(request.form['Loan_Amount_Term'])
    credit_history = float(request.form['Credit_History'])
    property_area = request.form['Property_Area']

    # Encode categorical values
    gender = 0 if gender == "Male" else 1
    married = 1 if married == "Yes" else 0
    education = 0 if education == "Graduate" else 1
    self_employed = 1 if self_employed == "Yes" else 0
    property_map = {"Urban":0, "Semiurban":1, "Rural":2}
    property_area = property_map[property_area]

    # Create dataframe
    input_data = pd.DataFrame([[gender, married, dependents, education,
                                self_employed, applicant_income,
                                coapplicant_income, loan_amount,
                                loan_term, credit_history, property_area]],
                              columns=['Gender','Married','Dependents','Education',
                                       'Self_Employed','ApplicantIncome','CoapplicantIncome',
                                       'LoanAmount','Loan_Amount_Term','Credit_History','Property_Area'])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    result = "Loan will be Approved ✅" if prediction == 1 else "Loan will NOT be Approved ❌"

    return render_template('submit.html', result=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True)
