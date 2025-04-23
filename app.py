from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model, scaler, and encoder
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))  # OneHotEncoder for Soil_Type

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Print all form data received (for debugging)
        print("Form data received:", request.form)

        # List of expected input fields
        input_fields = [
            'nickel', 'lead', 'chromium', 'mercury', 'cadmium',
            'arsenic', 'copper', 'zinc', 'sample_depth', 'latitude', 'longitude', 'soil_type'
        ]

        # Validate all fields are filled
        missing_fields = [field for field in input_fields if field not in request.form or request.form[field].strip() == '']
        if missing_fields:
            return render_template('index.html', error=f"Error: Missing input for {', '.join(missing_fields)}")

        # Extract numeric features
        numeric_features = np.array([
            float(request.form['nickel'].strip()),
            float(request.form['lead'].strip()),
            float(request.form['chromium'].strip()),
            float(request.form['mercury'].strip()),
            float(request.form['cadmium'].strip()),
            float(request.form['arsenic'].strip()),
            float(request.form['copper'].strip()),
            float(request.form['zinc'].strip()),
            float(request.form['sample_depth'].strip()),
            float(request.form['latitude'].strip()),
            float(request.form['longitude'].strip())
        ]).reshape(1, -1)  # Ensuring 2D array

        print("Numeric features shape:", numeric_features.shape)  # Should be (1, 11)

        # Encode soil type
        soil_type = [[request.form['soil_type'].strip()]]  # Needs to be 2D for encoder
        soil_type_encoded = encoder.transform(soil_type).toarray()

        print("Soil type encoded shape:", soil_type_encoded.shape)  # Should be (1, X), where X is soil type features

        # Combine numeric and categorical features
        final_features = np.hstack((numeric_features, soil_type_encoded))

        # Print the final shape before passing to model
        print("Final input shape before scaling:", final_features.shape)  # Should be (1, 14)

        # Scale the input features
        input_data = scaler.transform(final_features)

        print("Final input shape after scaling:", input_data.shape)  # Debugging

        # Make prediction
        prediction = model.predict(input_data)

        return render_template('index.html', prediction_text=f'Predicted Contamination Level: {prediction[0]}')

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
