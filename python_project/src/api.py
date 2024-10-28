# api.py
from flask import Flask, request, jsonify, Response
from flask_cors import CORS  # Add CORS support
from typing import Dict, Any, Optional
import pandas as pd
import joblib
import logging
from http import HTTPStatus

from sklearn.pipeline import Pipeline

from data_preprocessor import DataPreprocessor


FEATURE_NAMES = [
    'batteryPower', 'bluetooth', 'clockSpeed', 'dualSim', 'frontCamera', 
    'fourG', 'internalMemory', 'mobileDepth', 'mobileWeight', 'numCores', 
    'primaryCamera', 'pixelHeight', 'pixelWidth', 'ram', 'screenHeight', 
    'screenWidth', 'talkTime', 'threeG', 'touchScreen', 'wifi',
    'screenArea', 'pixelDensity', 'cameraRatio', 'memoryPerCore', 'batteryEfficiency'
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS

# Feature names and types validation
FEATURE_SCHEMA = {
    'batteryPower': int,
    'bluetooth': bool,
    'clockSpeed': float,
    'dualSim': bool,
    'frontCamera': int,
    'fourG': bool,
    'internalMemory': int,
    'mobileDepth': float,
    'mobileWeight': int,
    'numCores': int,
    'primaryCamera': int,
    'pixelHeight': int,
    'pixelWidth': int,
    'ram': int,
    'screenHeight': int,
    'screenWidth': int,
    'talkTime': int,
    'threeG': bool,
    'touchScreen': bool,
    'wifi': bool
}

def load_model() -> Pipeline:
    """Load the trained model"""
    try:
        model = joblib.load('device_price_model.pkl')
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def validate_input(data: Dict[str, Any]) -> Optional[str]:
    """Validate input data against schema"""
    if not data:
        return "No input data provided"
    
    for feature, expected_type in FEATURE_SCHEMA.items():
        if feature not in data:
            return f"Missing required feature: {feature}"
        
        value = data[feature]
        if expected_type == bool:
            if not isinstance(value, (bool, int)) or (isinstance(value, int) and value not in [0, 1]):
                return f"Invalid value for {feature}: must be boolean or 0/1"
        elif not isinstance(value, expected_type):
            return f"Invalid type for {feature}: expected {expected_type.__name__}"
            
        if expected_type in [int, float] and value < 0:
            return f"Invalid value for {feature}: must be non-negative"
    
    return None

def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered features required by the model."""
    # 1. Screen Area (with validation)
    df['screenArea'] = df['screenHeight'] * df['screenWidth']
    df.loc[df['screenArea'] == 0, 'screenArea'] = 1  # Prevent division by zero
    
    # 2. Pixel Density (with validation)
    total_pixels = df['pixelHeight'] * df['pixelWidth']
    df['pixelDensity'] = total_pixels / df['screenArea']
    
    # 3. Camera Ratio (with validation)
    df['cameraRatio'] = df.apply(lambda row: row['primaryCamera'] / row['frontCamera'] if row['frontCamera'] > 0 else row['primaryCamera'], axis=1)
    
    # 4. Memory per Core
    df['memoryPerCore'] = df['internalMemory'] / df['numCores'].replace(0, 1)
    
    # 5. Battery Efficiency (battery power per screen area)
    df['batteryEfficiency'] = df['batteryPower'] / df['screenArea']
    
    return df

@app.route('/health', methods=['GET'])
def health_check() -> Response:
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            logging.error("No input data provided")
            return jsonify({"error": "No input data provided"}), 400

        # Create DataFrame with expected features in correct order
        features = pd.DataFrame([{name: data.get(name) for name in FEATURE_SCHEMA.keys()}])
        
        # Validate all features are present
        validation_error = validate_input(data)
        if validation_error:
            logging.error(validation_error)
            return jsonify({"error": validation_error}), 400
        
        # Compute engineered features
        features = compute_engineered_features(features)

        # Load model
        model = load_model()
        
        # Make prediction using the pipeline
        prediction = model.predict(features)

        return jsonify({"priceRange": int(prediction[0])})

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict/bulk', methods=['POST'])
def predict_bulk() -> Response:
    """Bulk prediction endpoint"""
    try:
        data = request.get_json()
        logger.info(f"Received data for bulk prediction: {data}")

        if not isinstance(data, list):
            logger.error("Input data is not a list")
            return jsonify({
                "error": "Input must be a list of devices",
                "status": "error"
            }), HTTPStatus.BAD_REQUEST

        validated_data = []
        for idx, device in enumerate(data):
            error = validate_input(device)
            if error:
                logger.error(f"Validation error for device at index {idx}: {error}")
                return jsonify({
                    "error": f"Invalid device at index {idx}: {error}",
                    "status": "validation_error"
                }), HTTPStatus.BAD_REQUEST
            validated_data.append(device)

        # Convert boolean features to 0/1 for all devices
        for device in validated_data:
            for feature in ['bluetooth', 'dualSim', 'fourG', 'threeG', 'touchScreen', 'wifi']:
                device[feature] = 1 if device[feature] else 0

        # Create DataFrame
        features_df = pd.DataFrame(validated_data)
        logger.info(f"Features DataFrame for prediction: {features_df.head()}")

        # Compute engineered features
        features_df = compute_engineered_features(features_df)
        logger.info(f"Engineered features DataFrame: {features_df.head()}")

        # Check for NaN values
        if features_df.isnull().any().any():
            logger.error(f"DataFrame contains NaN values: {features_df.isnull().sum()}")
            return jsonify({
                "error": "Data contains NaN values after feature engineering",
                "status": "validation_error"
            }), HTTPStatus.BAD_REQUEST

        # Load model
        model = load_model()

        # Make predictions using the pipeline
        predictions = model.predict(features_df)
        logger.info(f"Predictions: {predictions}")

        return jsonify({
            "predictions": [int(pred) for pred in predictions],
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Error during bulk prediction: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "status": "error"
        }), HTTPStatus.INTERNAL_SERVER_ERROR


if __name__ == '__main__':
    # Load model on startup to validate it exists
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
