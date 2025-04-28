from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import sys

# Add path so utils can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

from predict import predict_failure

# Initialize Flask app
app = Flask(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'lightgbm_best_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return "Welcome to Predictive Maintenance API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = predict_failure(model, features)
    return jsonify({'failure_probability': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
