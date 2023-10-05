from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('car_price_predictor.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        present_price = float(request.form['present_price'])
        driven_kms = int(request.form['driven_kms'])
        fuel_type = request.form['fuel_type']
        selling_type = request.form['selling_type']
        transmission = request.form['transmission']
        owner = int(request.form['owner'])
        age = int(request.form['age'])

        # Preprocess the input data
        fuel_type_encoded = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}[fuel_type]
        selling_type_encoded = {'Dealer': 0, 'Individual': 1}[selling_type]
        transmission_encoded = {'Manual': 0, 'Automatic': 1}[transmission]

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Present_Price': [present_price],
            'Driven_kms': [driven_kms],
            'Fuel_Type': [fuel_type_encoded],
            'Selling_type': [selling_type_encoded],
            'Transmission': [transmission_encoded],
            'Owner': [owner],
            'Age': [age]
        })

        # Make a prediction using the model
        predicted_price = model.predict(input_data)[0]

        return render_template('index.html', prediction=f'Predicted Price: {predicted_price:.2f} Lakh INR')
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
