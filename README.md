# Devices Price Classification System

This project is an AI-based system built using Python and Spring Boot to classify the price of mobile devices based on their specifications. The system is divided into two parts:

- **Python Project**: Implements a machine learning model that predicts the price range of a device based on its specifications.
- **Spring Boot Project**: Contains a RESTful API that interacts with the Python model, stores device information, and provides endpoints for managing devices and getting price predictions.

## Table of Contents

1. [Setup Instructions](#setup-instructions)
2. [Running the Services](#running-the-services)
3. [Interacting with Endpoints](#interacting-with-endpoints)
4. [Code Design Decisions](#code-design-decisions)
5. [Testing](#testing)
6. [Data Preparation](#data-preparation)

## Setup Instructions

### Prerequisites

- Python 3.x
- Java 17
- Apache Maven
- PostgreSQL

### Python Project Setup

1. Create a virtual environment and activate it:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. Install dependencies from `requirements.txt`:

   ```sh
   pip install -r requirements.txt
   ```

3. Train the machine learning model and start the Flask API:

   ```sh
   python model.py  # To train the model
   python api.py    # To run the Flask API
   ```

### Spring Boot Project Setup

1. Install dependencies using Maven:

   ```sh
   mvn clean install
   ```

2. Set up the PostgreSQL database:

   - Create a database named `device_pricing`.
   - Update the `application.properties` file with your PostgreSQL credentials.

3. Run the Spring Boot application:

   ```sh
   mvn spring-boot:run
   ```

## Running the Services

- **Python Flask API** will be available at: `http://localhost:5000`
- **Spring Boot Application** will be available at: `http://localhost:8080/api`

## Interacting with Endpoints

### Device Management Endpoints

1. **Create a New Device**

   - Endpoint: `POST /api/devices`
   - Request Body:
     ```json
     {
       "battery_power": 3500,
       "blue": 1,
       "clock_speed": 2.2,
       "dual_sim": 1,
       "fc": 8,
       "four_g": 1,
       "int_memory": 64,
       "m_dep": 0.8,
       "mobile_wt": 150,
       "n_cores": 4,
       "pc": 12,
       "px_height": 1800,
       "px_width": 1080,
       "ram": 4000,
       "sc_h": 14,
       "sc_w": 7,
       "talk_time": 24,
       "three_g": 1,
       "touch_screen": 1,
       "wifi": 1
     }
     ```

2. **Get Device by ID**

   - Endpoint: `GET /api/devices/{id}`

3. **Update an Existing Device**

   - Endpoint: `PUT /api/devices/{id}`

4. **Delete a Device**

   - Endpoint: `DELETE /api/devices/{id}`

### Price Prediction Endpoints

1. **Predict Price for a Device**

   - Endpoint: `POST /api/devices/predict/{deviceId}`

2. **Bulk Prediction**

   - Endpoint: `POST /api/devices/predict/bulk`
   - Request Body: An array of device specifications (similar to the single device creation format).

## Code Design Decisions

- **Model Choice**: The `RandomForestClassifier` was chosen due to its robustness in handling both numerical and categorical features and its ability to prevent overfitting with hyperparameter tuning.
- **Hyperparameter Tuning**: Techniques like `GridSearchCV` were used to select optimal parameters for the Random Forest model.
- **Data Preparation**: Missing values were handled by filling them with the mean value of each column, ensuring consistency in training.

## Testing

- **Unit and Integration Testing**: Unit tests are provided for individual service methods. Integration tests are used to validate the interaction between the Spring Boot application and the Python Flask API.
- **End-to-End Testing**: Run predictions for 10 devices from `test.csv` using the `run_bulk_predictions.py` script to validate the system.

## Data Preparation

- **Feature Engineering**: Created new features by combining correlated features to enhance model performance.
- **EDA (Exploratory Data Analysis)**: Visualizations, such as histograms and correlation heatmaps, were generated to understand feature distributions and correlations.
- **Outlier Detection**: Outliers were identified and removed to improve model stability.

## Usage Examples

- **Running Bulk Predictions**: Use the `run_bulk_predictions.py` script to send bulk requests to the `/predict/bulk` endpoint.

  ```sh
  python run_bulk_predictions.py
  ```

- **Automated Testing**: Run the `end_to_end_test.py` script to validate the end-to-end functionality of the system.

  ```sh
  python end_to_end_test.py
  ```

## Environment Setup Guide

- **Python Virtual Environment**: To avoid dependency conflicts, create and activate a Python virtual environment as described in the setup instructions.
- **PostgreSQL Configuration**: Install PostgreSQL and create the required database (`device_pricing`). Ensure that the credentials match those configured in the Spring Boot project's `application.properties`.
- **Dependencies**: Use the provided `requirements.txt` for Python dependencies and Maven for Java dependencies.

## Design Decisions Justification

- **Random Forest vs. Other Models**: Random Forest was chosen after experimenting with other classifiers like Logistic Regression and SVM. Random Forest provided the highest accuracy and showed robustness to overfitting due to the ensemble nature of the model. SVM and Logistic Regression, while useful, did not handle the feature variety as effectively.
- **Data Preprocessing**: Missing values were filled with the mean to maintain consistency without significantly altering feature distributions. The dataset had a few outliers that were removed to improve overall model stability.
- **Transaction Management**: The `/predict/{deviceId}` endpoint is annotated with `@Transactional` to ensure consistency between prediction and database updates, especially in case of errors during external API calls.

## Model Results

- Best parameters: {'classifier__max_depth': None, 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 2, 'classifier__n_estimators': 300}
- Best cross-validation score: 0.873
- Model training completed successfully
- Model evaluation completed
- Accuracy: 0.902


- **Final Model Evaluation Results:**
- Accuracy: 0.902
- F1 Score: 0.902

- **Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.96      | 0.98   | 0.97     | 100     |
| 1     | 0.87      | 0.85   | 0.86     | 100     |
| 2     | 0.84      | 0.83   | 0.83     | 100     |
| 3     | 0.94      | 0.95   | 0.95     | 100     |

| Metric        | Value |
|---------------|-------|
| Accuracy      | 0.90  |
| Macro Avg     | 0.90  |
| Weighted Avg  | 0.90  |

- **Feature Importance:**

- ram: 0.4661
- batteryPower: 0.0620
- pixelWidth: 0.0469
- pixelHeight: 0.0434
- mobileWeight: 0.0313
- batteryEfficiency: 0.0311
- pixelDensity: 0.0302
- memoryPerCore: 0.0295
- internalMemory: 0.0260
- cameraRatio: 0.0257

## Future Improvements

- **Enhanced Model Tuning**: Use more sophisticated hyperparameter tuning methods, such as Bayesian optimization, to further enhance model accuracy.
- **User Interface**: Develop a simple front-end UI to interact with the API for non-technical users.
- **Deployment**: Consider containerizing both the Python and Spring Boot services using Docker to simplify deployment and scaling.


MIT License

Copyright (c) 2024 Ahmad

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Contact Information

For further details or questions about the project, please contact:

- **Developer**: Ahmad Konainah
- **Email**: [ahmad.konainah@gmail.com](mailto\:ahmad.konainah@gmail.com)

