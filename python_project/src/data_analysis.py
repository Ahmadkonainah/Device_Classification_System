import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    def __init__(self, data_path='train.csv'):
        self.data_path = data_path
        self.data = None
        
    def load_data(self):
        try:
            self.data = pd.read_csv(self.data_path)
            
            # Convert column names to camelCase
            self.data = self.data.rename(columns={
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
            })
            
            # Handle missing values
            self.data = self.data.fillna(self.data.mean())
            
            logger.info("Data loaded and columns renamed successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False

    def generate_feature_distributions(self):
        try:
            # Create directory for plots if it doesn't exist
            import os
            os.makedirs('plots', exist_ok=True)

            # Plot histograms for numerical features
            plt.figure(figsize=(20, 15))
            self.data.hist(bins=30)
            plt.tight_layout()
            plt.savefig('plots/feature_distributions.png')
            plt.close()

            logger.info("Feature distribution plots generated successfully")
            return True
        except Exception as e:
            logger.error(f"Error generating feature distributions: {str(e)}")
            return False

    def analyze_feature_importance(self):
        try:
            # Separate features and target
            X = self.data.drop('priceRange', axis=1)
            y = self.data['priceRange']

            # Calculate mutual information scores
            mi_importance = mutual_info_classif(X, y)
            
            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': mi_importance
            }).sort_values('importance', ascending=False)

            # Plot feature importance
            plt.figure(figsize=(12, 6))
            sns.barplot(x='importance', y='feature', data=importance_df)
            plt.title('Feature Importance using Mutual Information')
            plt.savefig('plots/feature_importance.png')
            plt.close()

            logger.info("Feature importance analysis completed successfully")
            return importance_df
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {str(e)}")
            return None

    def analyze_correlations(self):
        try:
            # Calculate correlations
            correlations = self.data.corr()

            # Plot correlation heatmap
            plt.figure(figsize=(15, 12))
            sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlations')
            plt.savefig('plots/correlation_matrix.png')
            plt.close()

            logger.info("Correlation analysis completed successfully")
            return correlations
        except Exception as e:
            logger.error(f"Error analyzing correlations: {str(e)}")
            return None

    def run_analysis(self):
        try:
            if not self.load_data():
                raise Exception("Failed to load data")

            # Run all analyses
            self.generate_feature_distributions()
            importance_df = self.analyze_feature_importance()
            correlations = self.analyze_correlations()

            # Print summary statistics
            print("\nDataset Summary:")
            print("-" * 50)
            print(f"Number of samples: {len(self.data)}")
            print(f"Number of features: {len(self.data.columns) - 1}")  # Excluding target
            print("\nFeature Importance Summary:")
            if importance_df is not None:
                print(importance_df)

            print("\nMissing Values Summary:")
            print(self.data.isnull().sum())

        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    analyzer = DataAnalyzer()
    analyzer.run_analysis()