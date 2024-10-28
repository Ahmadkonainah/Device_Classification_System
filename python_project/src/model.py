import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import logging
from typing import Dict, Any
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Change Matplotlib backend to avoid Tkinter issues
matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    'batteryPower', 'bluetooth', 'clockSpeed', 'dualSim', 'frontCamera', 
    'fourG', 'internalMemory', 'mobileDepth', 'mobileWeight', 'numCores', 
    'primaryCamera', 'pixelHeight', 'pixelWidth', 'ram', 'screenHeight', 
    'screenWidth', 'talkTime', 'threeG', 'touchScreen', 'wifi'
]

class DevicePriceModel:
    def __init__(self, random_state: int = 42):
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=200,  # Increased from 100
                max_depth=None,    # Allow full depth
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=-1  # Use all CPU cores
            ))
        ])
        
        self.feature_importance: Dict[str, float] = {}
        self.model_metrics: Dict[str, float] = {}
        self.is_fitted = False

    def perform_hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series) -> None:
        logger.info("Starting hyperparameter tuning...")
        
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring='accuracy'
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        
        # Update the model with best parameters
        self.pipeline.set_params(**grid_search.best_params_)

    def analyze_features(self, X: pd.DataFrame, y: pd.Series) -> None:
        logger.info("Performing feature analysis...")
        
        # Correlation analysis
        correlation_matrix = X.corr()
        plt.figure(figsize=(15, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Heatmap')
        plt.savefig('correlation_heatmap.png')
        plt.close()
        
        # Feature distributions by price range
        for column in X.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=y, y=X[column])
            plt.title(f'{column} Distribution by Price Range')
            plt.savefig(f'feature_dist_{column}.png')
            plt.close()

    def train(self, X: pd.DataFrame, y: pd.Series, perform_tuning: bool = True) -> None:
        try:
            logger.info("Starting model training...")
            
            # Perform feature analysis
            self.analyze_features(X, y)
            
            if perform_tuning:
                self.perform_hyperparameter_tuning(X, y)
            
            # Train the model
            self.pipeline.fit(X, y)
            self.is_fitted = True
            
            # Get feature importance from the Random Forest classifier
            feature_importance = self.pipeline.named_steps['classifier'].feature_importances_
            self.feature_importance = dict(zip(X.columns, feature_importance))
            
            # Sort features by importance
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Plot feature importance
            plt.figure(figsize=(12, 6))
            features, importance = zip(*sorted_features)
            plt.bar(features, importance)
            plt.xticks(rotation=45, ha='right')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            plt.close()
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before evaluation")
        
        try:
            predictions = self.pipeline.predict(X)
            
            self.model_metrics = {
                'accuracy': accuracy_score(y, predictions),
                'f1_score': f1_score(y, predictions, average='weighted'),
                'classification_report': classification_report(y, predictions),
                'confusion_matrix': confusion_matrix(y, predictions)
            }
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(self.model_metrics['confusion_matrix'], 
                       annot=True, 
                       fmt='d', 
                       cmap='Blues')
            plt.title('Confusion Matrix')
            plt.savefig('confusion_matrix.png')
            plt.close()
            
            logger.info("Model evaluation completed")
            logger.info(f"Accuracy: {self.model_metrics['accuracy']:.3f}")
            
            return self.model_metrics
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before prediction")
            
        try:
            return self.pipeline.predict(X)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

    def save_model(self, model_path: str = 'device_price_model.pkl', scaler_path: str = 'scaler.pkl') -> None:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before saving")
            
        try:
            logger.info(f"Saving model to {model_path}")
            import joblib
            
            # Save the entire pipeline (includes both scaler and model)
            joblib.dump(self.pipeline, model_path)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Load the data
        logger.info("Loading data...")
        data = pd.read_csv('train.csv')
        
        # Log initial data info
        logger.info(f"Initial data shape: {data.shape}")
        logger.info("\nMissing values summary:")
        logger.info(data.isnull().sum())
        
        # Rename columns
        column_mapping = {
            'battery_power': 'batteryPower',
            'blue': 'bluetooth',
            'clock_speed': 'clockSpeed',
            'dual_sim': 'dualSim',
            'fc': 'frontCamera',
            'four_g': 'fourG',
            'int_memory': 'internalMemory',
            'm_dep': 'mobileDepth',
            'mobile_wt': 'mobileWeight',
            'n_cores': 'numCores',
            'pc': 'primaryCamera',
            'px_height': 'pixelHeight',
            'px_width': 'pixelWidth',
            'ram': 'ram',
            'sc_h': 'screenHeight',
            'sc_w': 'screenWidth',
            'talk_time': 'talkTime',
            'three_g': 'threeG',
            'touch_screen': 'touchScreen',
            'wifi': 'wifi',
            'price_range': 'priceRange'
        }
        data = data.rename(columns=column_mapping)
        
        # Feature engineering with safety checks
        logger.info("Performing feature engineering...")
        
        # 1. Screen Area (with validation)
        data['screenArea'] = data['screenHeight'] * data['screenWidth']
        data.loc[data['screenArea'] == 0, 'screenArea'] = 1  # Prevent division by zero
        
        # 2. Pixel Density (with validation)
        total_pixels = data['pixelHeight'] * data['pixelWidth']
        data['pixelDensity'] = np.where(
            data['screenArea'] > 0,
            total_pixels / data['screenArea'],
            0
        )
        
        # 3. Camera Ratio (with validation)
        data['cameraRatio'] = np.where(
            data['frontCamera'] > 0,
            data['primaryCamera'] / data['frontCamera'],
            data['primaryCamera']
        )
        
        # 4. Memory per Core
        data['memoryPerCore'] = data['internalMemory'] / data['numCores'].replace(0, 1)
        
        # 5. Battery Efficiency (battery power per screen area)
        data['batteryEfficiency'] = data['batteryPower'] / data['screenArea'].replace(0, 1)
        
        # Remove any infinite values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median
        data = data.fillna(data.median())
        
        # Log engineered features statistics
        logger.info("\nEngineered features statistics:")
        logger.info(data.describe())
        
        # Split features and target
        X = data.drop('priceRange', axis=1)
        y = data['priceRange']
        
        # Check for any remaining infinite values
        if np.any(np.isinf(X.values)):
            logger.error("Infinite values found in features!")
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize model
        logger.info("Initializing model...")
        model = DevicePriceModel()
        
        # Train the model
        logger.info("Training model with initial parameters...")
        model.train(X_train, y_train, perform_tuning=True)
        
        # Evaluate the model
        evaluation_results = model.evaluate(X_test, y_test)
        
        # Print results
        logger.info("\nFinal Model Evaluation Results:")
        logger.info(f"Accuracy: {evaluation_results['accuracy']:.3f}")
        logger.info(f"F1 Score: {evaluation_results['f1_score']:.3f}")
        logger.info("\nClassification Report:")
        logger.info(evaluation_results['classification_report'])
        
        # Feature importance
        logger.info("\nFeature Importance:")
        for feature, importance in sorted(
            model.feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]:  # Show top 10 features
            logger.info(f"{feature}: {importance:.4f}")
        
        # Save the model and scaler
        model.save_model()
        logger.info("Model has been saved to disk")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
