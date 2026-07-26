import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("loan_prediction.csv")

# Encode categorical columns
data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})
data["Married"] = data["Married"].map({"No": 0, "Yes": 1})
data["Dependents"] = data["Dependents"].replace("3+", 3).astype(int)
data["Education"] = data["Education"].map({"Graduate": 0, "Not Graduate": 1})
data["Self_Employed"] = data["Self_Employed"].map({"No": 0, "Yes": 1})
data["Property_Area"] = data["Property_Area"].map({"Urban": 0, "Semiurban": 1, "Rural": 2})

# Select features and target
X = data[["Gender", "Married", "Dependents", "Education", "Self_Employed",
          "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
          "Loan_Amount_Term", "Credit_History", "Property_Area"]]
y = data["Loan_Status"].map({"Y": 1, "N": 0})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model and scaler (names match what app.py expects)
pickle.dump(model, open("rdf.pkl", "wb"))
pickle.dump(scaler, open("scale1.pkl", "wb"))

print("Model trained and saved as rdf.pkl and scale1.pkl")
