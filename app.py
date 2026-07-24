import numpy as np
import pickle
import pandas
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
    # Read inputs from form
    input_feature = [float(x) for x in request.form.values()]
    input_feature = np.array([input_feature])
    names = ['Gender', 'Married', 'Dependents', 'Education', 'Self Employed',
             'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
             'Loan_Amount_Term', 'Credit_History', 'Property_Area']
    data = pandas.DataFrame(input_feature, columns=names)

    # Predict using model
    prediction = model.predict(data)
    prediction = int(prediction)

    # Show result
    if prediction == 0:
        result = "Loan will NOT be Approved ❌"
    else:
        result = "Loan will be Approved ✅"

    return render_template('submit.html', result=result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False)
