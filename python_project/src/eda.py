# eda.py - Enhanced version

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List
import logging
from pathlib import Path
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings but store them for later review if needed
warnings.filterwarnings('ignore')
stored_warnings = []
warnings.showwarning = lambda message, *args: stored_warnings.append(str(message))

class DeviceDataAnalyzer:
    def __init__(self, data_path: str):
        """Initialize the analyzer with data path."""
        self.data_path = data_path
        self.data = None
        self.numeric_features = None
        self.categorical_features = None
        
    def load_data(self) -> None:
        """Load and prepare the dataset."""
        try:
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded data from {self.data_path}")
            
            # Convert column names to camelCase
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
            self.data = self.data.rename(columns=column_mapping)
            
            # Identify numeric and categorical features
            self.numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns
            self.categorical_features = self.data.select_dtypes(include=['bool']).columns
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def generate_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics for numeric features."""
        summary = self.data[self.numeric_features].describe()
        logger.info("\nSummary Statistics:")
        print(summary)
        return summary

    def analyze_missing_values(self) -> pd.DataFrame:
        """Analyze missing values in the dataset."""
        missing_values = self.data.isnull().sum()
        missing_percentage = (missing_values / len(self.data)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percentage
        })
        logger.info("\nMissing Values Analysis:")
        print(missing_df[missing_df['Missing Values'] > 0])
        return missing_df

    def plot_distributions(self, output_dir: str = 'plots') -> None:
        """Plot distributions for all numeric features."""
        Path(output_dir).mkdir(exist_ok=True)
        
        for feature in self.numeric_features:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=self.data, x=feature, hue='priceRange', multiple="stack")
            plt.title(f'Distribution of {feature} by Price Range')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{feature}_distribution.png')
            plt.close()
            
        logger.info(f"Distribution plots saved in {output_dir}")

    def plot_correlation_matrix(self, output_dir: str = 'plots') -> None:
        """Generate and save correlation matrix heatmap."""
        Path(output_dir).mkdir(exist_ok=True)
        
        plt.figure(figsize=(15, 12))
        correlation_matrix = self.data[self.numeric_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_matrix.png')
        plt.close()
        
        logger.info("Correlation matrix plot saved")

    def analyze_price_distribution(self) -> None:
        """Analyze the distribution of price ranges."""
        price_dist = self.data['priceRange'].value_counts().sort_index()
        plt.figure(figsize=(10, 6))
        price_dist.plot(kind='bar')
        plt.title('Distribution of Price Ranges')
        plt.xlabel('Price Range')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('plots/price_distribution.png')
        plt.close()
        
        logger.info("\nPrice Range Distribution:")
        print(price_dist)

    def identify_outliers(self) -> pd.DataFrame:
        """Identify outliers in numeric features using IQR method."""
        outliers_summary = {}
        
        for feature in self.numeric_features:
            Q1 = self.data[feature].quantile(0.25)
            Q3 = self.data[feature].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.data[feature] < (Q1 - 1.5 * IQR)) | 
                       (self.data[feature] > (Q3 + 1.5 * IQR))).sum()
            outliers_summary[feature] = {
                'count': outliers,
                'percentage': (outliers / len(self.data)) * 100
            }
            
        outliers_df = pd.DataFrame(outliers_summary).T
        logger.info("\nOutliers Summary:")
        print(outliers_df)
        return outliers_df

def main():
    # Initialize analyzer
    analyzer = DeviceDataAnalyzer('train.csv')
    
    # Create analysis pipeline
    try:
        analyzer.load_data()
        analyzer.generate_summary_statistics()
        analyzer.analyze_missing_values()
        analyzer.plot_distributions()
        analyzer.plot_correlation_matrix()
        analyzer.analyze_price_distribution()
        analyzer.identify_outliers()
        
        if stored_warnings:
            logger.warning("The following warnings were suppressed during analysis:")
            for warning in stored_warnings:
                logger.warning(warning)
                
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()