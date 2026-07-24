from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/result', methods=['POST'])
def result():
    # Collect form data
    name = request.form['name']
    age = request.form['age']
    income = request.form['income']

    # Dummy prediction logic (replace with ML model later)
    if int(income) > 50000:
        prediction = "Loan Approved ✅"
    else:
        prediction = "Loan Not Approved ❌"

    return render_template('output.html', name=name, age=age, income=income, prediction=prediction)

if __name__ == '__main__':
     app.run(host="0.0.0.0", port=8000, debug=True)
