# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv("Loan_Data.csv")

# Drop irrelevant column
data.drop(columns=["Loan_ID"], inplace=True)

# Handle missing values
data.fillna(data.mean(numeric_only=True), inplace=True)

# Encode categorical variables
categorical_columns = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area", "Loan_Status"]
encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = encoder.fit_transform(data[column].astype(str))

# Features and target variable
X = data.drop(columns=["Loan_Status"])
y = data["Loan_Status"]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Save the model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
