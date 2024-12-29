from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form
    input_features = [
        int(form_data['Gender']),
        int(form_data['Married']),
        int(form_data['Dependents']),
        int(form_data['Education']),
        int(form_data['Self_Employed']),
        float(form_data['ApplicantIncome']),
        float(form_data['CoapplicantIncome']),
        float(form_data['LoanAmount']),
        float(form_data['Loan_Amount_Term']),
        int(form_data['Credit_History']),
        int(form_data['Property_Area']),
    ]

    # Scale input features and make prediction
    new_data_scaled = scaler.transform([input_features])
    prediction = model.predict(new_data_scaled)
    result = "Approved" if prediction[0] == 1 else "Not Approved"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
