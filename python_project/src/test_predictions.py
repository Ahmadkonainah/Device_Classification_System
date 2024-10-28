import requests
import time
import pandas as pd
import random

# Define the required fields
REQUIRED_FIELDS = [
    'batteryPower', 'bluetooth', 'clockSpeed', 'dualSim', 'frontCamera',
    'fourG', 'internalMemory', 'mobileDepth', 'mobileWeight', 'numCores',
    'primaryCamera', 'pixelHeight', 'pixelWidth', 'ram', 'screenHeight',
    'screenWidth', 'talkTime', 'threeG', 'touchScreen', 'wifi'
]

# Define value ranges for randomization
VALUE_RANGES = {
    'batteryPower': (500, 5000),
    'bluetooth': [True, False],
    'clockSpeed': (0.5, 3.5),
    'dualSim': [True, False],
    'frontCamera': (0, 20),
    'fourG': [True, False],
    'internalMemory': (8, 256),
    'mobileDepth': (0.1, 1.0),
    'mobileWeight': (80, 300),
    'numCores': (1, 8),
    'primaryCamera': (0, 40),
    'pixelHeight': (480, 2160),
    'pixelWidth': (480, 2160),
    'ram': (512, 8192),
    'screenHeight': (3, 20),
    'screenWidth': (2, 10),
    'talkTime': (1, 30),
    'threeG': [True, False],
    'touchScreen': [True, False],
    'wifi': [True, False]
}

def get_random_value(field):
    """Returns a random value for the given field based on its predefined range."""
    if isinstance(VALUE_RANGES[field], tuple):
        if field == 'mobileDepth':
            return round(random.uniform(*VALUE_RANGES[field]), 2)  # Keep mobileDepth as float with two decimal places
        else:
            return random.randint(int(VALUE_RANGES[field][0]), int(VALUE_RANGES[field][1]))
    elif isinstance(VALUE_RANGES[field], list):
        return random.choice(VALUE_RANGES[field])

def test_predictions():
    print("\nTesting predictions for 10 devices:\n")
    
    # Load test data from CSV
    test_data = pd.read_csv('test.csv')
    
    # Randomly select 10 rows from the test data and make sure all required fields are present
    devices = test_data.sample(n=10, random_state=42).to_dict(orient='records')
    
    # Ensure all required fields are present and have valid positive values
    for device in devices:
        # Remove any existing 'id' field from the device data
        if 'id' in device:
            del device['id']
        # Check if all required fields are valid and replace invalid ones with randomized values
        for field in REQUIRED_FIELDS:
            if field not in device or device[field] <= 0:
                device[field] = get_random_value(field)

    created_device_ids = []

    for i, device_data in enumerate(devices):
        try:
            # Create device
            create_response = requests.post(
                "http://localhost:8080/api/devices",
                json=device_data,
                headers={"Content-Type": "application/json"}
            )

            if create_response.status_code == 201:
                print(f"Device {i} created successfully.")
                device_id = create_response.json().get("id")
                if device_id:
                    created_device_ids.append(device_id)
            else:
                print(f"Failed to create device {i}: {create_response.status_code}, {create_response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to create device {i}: {e}")

        # Pause between requests to avoid overwhelming the server
        time.sleep(1)

    # Predict for each created device
    for i, device_id in enumerate(created_device_ids):
        try:
            predict_response = requests.post(
                f"http://localhost:8080/api/devices/predict/{device_id}"
            )

            if predict_response.status_code == 200:
                response_json = predict_response.json()
                print(f"Full response for device {i}: {response_json}")
                price_range = response_json.get("priceRange")
                print(f"Prediction for device {i}: {price_range}")
            else:
                print(f"Failed to get prediction for device {i}: {predict_response.status_code}, {predict_response.text}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to get prediction for device {i}: {e}")

if __name__ == "__main__":
    test_predictions()
