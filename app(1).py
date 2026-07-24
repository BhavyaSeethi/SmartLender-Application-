
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("loan_prediction.csv")

# Select features and target
X = data[["Gender","Married","Dependents","Education","Self_Employed",
          "ApplicantIncome","CoapplicantIncome","LoanAmount",
          "Loan_Amount_Term","Credit_History","Property_Area"]]
y = data["Loan_Status"].map({"Y":1,"N":0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scale.pkl", "wb"))
