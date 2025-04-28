import joblib
import numpy as np

# Load the model
model = joblib.load('app/model/lightgbm_best_model.pkl')

def predict_failure(features):
    """
    Predicts failure based on input features.

    Args:
        features (list or array): input sensor values.

    Returns:
        int: 0 for no failure, 1 for failure.
    """
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]
