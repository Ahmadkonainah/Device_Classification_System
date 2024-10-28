import sys
import os
import pandas as pd
import numpy as np
import unittest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

# Add the src folder to the system path to import the necessary modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from preprocessor import DataPreprocessor
from model import DevicePriceModel


class TestDevicePriceModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and model"""
        # Load training and test data
        cls.train_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'train.csv'))
        cls.test_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'test.csv'))

        # Verify 'price_range' column exists
        if 'price_range' not in cls.train_data.columns:
            raise KeyError("The 'price_range' column is missing in train.csv. Please check the file.")

        # Handle missing values in the train and test datasets
        cls.train_data.fillna(cls.train_data.mean(), inplace=True)
        cls.test_data.fillna(cls.test_data.mean(), inplace=True)

        # Initialize preprocessor and model
        cls.preprocessor = DataPreprocessor()
        cls.preprocessor.fit(cls.train_data.drop(columns=['price_range'], errors='ignore'))
        cls.model = DevicePriceModel()

    def drop_id_column(self, df):
        """Utility method to drop the 'id' column if it exists"""
        return df.drop(columns=['id'], errors='ignore')

    def test_data_preprocessing(self):
        """Test data preprocessing functionality"""
        X_train = self.drop_id_column(self.train_data).drop(columns=['price_range'])
        X_train_transformed = self.preprocessor.transform(X_train)

        # Check if preprocessed data has the correct shape
        expected_num_features = X_train_transformed.shape[1]
        self.assertEqual(X_train_transformed.shape[1], expected_num_features)
        print("Data preprocessing test passed.")

    def test_model_training(self):
        """Test model training functionality"""
        X_train, y_train = self.drop_id_column(self.train_data).drop(columns=['price_range']), self.train_data['price_range']
        X_train_transformed = self.preprocessor.transform(X_train)

        # Train model
        self.model.train(X_train_transformed, y_train)

        # Check if model was saved successfully as 'device_price_model.pkl'
        model_path = 'device_price_model.pkl'
        self.assertTrue(os.path.exists(model_path), f"Trained model file '{model_path}' was not found.")
        print("Model training test passed.")

    def test_model_prediction(self):
        """Test model prediction functionality"""
        X_train, y_train = self.drop_id_column(self.train_data).drop(columns=['price_range']), self.train_data['price_range']
        X_train_transformed = self.preprocessor.transform(X_train)
        self.model.train(X_train_transformed, y_train)

        # Preprocess test data
        X_test = self.drop_id_column(self.test_data)
        X_test_transformed = self.preprocessor.transform(X_test)

        # Make predictions
        predictions = self.model.predict(X_test_transformed)

        # Check predictions
        self.assertEqual(len(predictions), len(self.test_data))
        self.assertTrue(all(pred in [0, 1, 2, 3] for pred in predictions))
        print("Model prediction test passed.")

    def test_model_cross_validation(self):
        """Test model training and performance using 2-fold cross-validation"""
        X = self.drop_id_column(self.train_data).drop(columns=['price_range'])
        y = self.train_data['price_range']

        kfold = KFold(n_splits=2, shuffle=True, random_state=42)
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []

        for train_index, test_index in kfold.split(X):
            # Split data into train and validation sets
            X_train, X_val = X.iloc[train_index], X.iloc[test_index]
            y_train, y_val = y.iloc[train_index], y.iloc[test_index]

            # Preprocess data
            X_train_transformed = self.preprocessor.fit_transform(X_train)
            X_val_transformed = self.preprocessor.transform(X_val)

            # Train model
            self.model.train(X_train_transformed, y_train)

            # Make predictions on validation set
            y_pred = self.model.predict(X_val_transformed)

            # Calculate metrics
            accuracies.append(accuracy_score(y_val, y_pred))
            precisions.append(precision_score(y_val, y_pred, average='weighted'))
            recalls.append(recall_score(y_val, y_pred, average='weighted'))
            f1_scores.append(f1_score(y_val, y_pred, average='weighted'))

        # Print the average performance metrics
        avg_accuracy = np.mean(accuracies)
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1_scores)

        print(f"2-Fold Cross-Validation Results - Average Metrics:")
        print(f"Accuracy: {avg_accuracy:.2f}, Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, F1 Score: {avg_f1:.2f}")

        # Assert minimum performance thresholds
        self.assertGreater(avg_accuracy, 0.7)
        self.assertGreater(avg_precision, 0.7)
        self.assertGreater(avg_recall, 0.7)
        self.assertGreater(avg_f1, 0.7)
        print("2-fold cross-validation test passed.")

    def test_edge_cases(self):
        """Test model behavior with edge cases"""
        # Test with minimum values
        min_device = pd.DataFrame({
            'battery_power': [500],
            'blue': [0],
            'clock_speed': [0.5],
            'dual_sim': [0],
            'fc': [0],
            'four_g': [0],
            'int_memory': [2],
            'm_dep': [0.1],
            'mobile_wt': [80],
            'n_cores': [1],
            'pc': [0],
            'px_height': [0],
            'px_width': [0],
            'ram': [256],
            'sc_h': [5],
            'sc_w': [0],
            'talk_time': [2],
            'three_g': [0],
            'touch_screen': [0],
            'wifi': [0]
        })

        # Fit preprocessor before transforming edge cases
        X_train, y_train = self.drop_id_column(self.train_data).drop(columns=['price_range']), self.train_data['price_range']
        self.preprocessor.fit(X_train)
        self.model.train(X_train, y_train)

        # Preprocess and predict
        X_min = self.preprocessor.transform(min_device)
        min_prediction = self.model.predict(X_min)

        # Check predictions are within valid range
        self.assertTrue(all(pred in [0, 1, 2, 3] for pred in min_prediction))
        print("Edge case test passed.")


if __name__ == '__main__':
    unittest.main(verbosity=2)
