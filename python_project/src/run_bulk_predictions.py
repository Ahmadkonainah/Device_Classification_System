# run_bulk_predictions.py

import pandas as pd
import requests
import logging
from typing import List, Dict, Any
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bulk_predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Feature names in the exact order as expected by the model
FEATURE_NAMES = ['id','batteryPower', 'bluetooth', 'clockSpeed', 'dualSim', 'frontCamera', 
    'fourG', 'internalMemory', 'mobileDepth', 'mobileWeight', 'numCores', 
    'primaryCamera', 'pixelHeight', 'pixelWidth', 'ram', 'screenHeight', 
    'screenWidth', 'talkTime', 'threeG', 'touchScreen', 'wifi',
    'screenArea', 'pixelDensity', 'cameraRatio', 'memoryPerCore', 'batteryEfficiency'
]

class BulkPredictionRunner:
    def __init__(self, 
                 api_url: str = "http://localhost:8080/api",
                 batch_size: int = 50,
                 max_workers: int = 4):
        self.api_url = api_url
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.results = []
        self.errors = []

    def prepare_device_data(self, row: pd.Series) -> Dict[str, Any]:
        """Convert DataFrame row to API-compatible format."""
        return {
            "id": int(row["id"]),
            "batteryPower": int(row["batteryPower"]),
            "bluetooth": bool(row["bluetooth"]),
            "clockSpeed": float(row["clockSpeed"]),
            "dualSim": bool(row["dualSim"]),
            "frontCamera": int(row["frontCamera"]),
            "fourG": bool(row["fourG"]),
            "internalMemory": int(row["internalMemory"]),
            "mobileDepth": float(row["mobileDepth"]),
            "mobileWeight": int(row["mobileWeight"]),
            "numCores": int(row["numCores"]),
            "primaryCamera": int(row["primaryCamera"]),
            "pixelHeight": int(row["pixelHeight"]),
            "pixelWidth": int(row["pixelWidth"]),
            "ram": int(row["ram"]),
            "screenHeight": int(row["screenHeight"]),
            "screenWidth": int(row["screenWidth"]),
            "talkTime": int(row["talkTime"]),
            "threeG": bool(row["threeG"]),
            "touchScreen": bool(row["touchScreen"]),
            "wifi": bool(row["wifi"]),
            "screenArea": float(row["screenArea"]),
            "pixelDensity": float(row["pixelDensity"]),
            "cameraRatio": float(row["cameraRatio"]),
            "memoryPerCore": float(row["memoryPerCore"]),
            "batteryEfficiency": float(row["batteryEfficiency"])
        }

    def compute_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute engineered features required by the model with detailed logging."""
        
        # Screen Area (with validation)
        df['screenArea'] = df['screenHeight'] * df['screenWidth']
        df.loc[df['screenArea'] == 0, 'screenArea'] = 1  # Prevent division by zero

        # Log any NaN values in 'screenArea'
        if df['screenArea'].isna().any():
            logger.error("NaN detected in 'screenArea'. Rows with NaN: \n%s", df[df['screenArea'].isna()])
        
        # Pixel Density (with validation)
        total_pixels = df['pixelHeight'] * df['pixelWidth']
        df['pixelDensity'] = total_pixels / df['screenArea']
        df['pixelDensity'].fillna(0, inplace=True)  # Replace NaN values with 0
        
        if df['pixelDensity'].isna().any():
            logger.error("NaN detected in 'pixelDensity'. Rows with NaN: \n%s", df[df['pixelDensity'].isna()])
        
        # Camera Ratio (with validation)
        df['cameraRatio'] = df.apply(lambda row: row['primaryCamera'] / row['frontCamera'] 
                                    if row['frontCamera'] > 0 else row['primaryCamera'], axis=1)
        df['cameraRatio'].fillna(0, inplace=True)

        if df['cameraRatio'].isna().any():
            logger.error("NaN detected in 'cameraRatio'. Rows with NaN: \n%s", df[df['cameraRatio'].isna()])
        
        # Memory per Core
        df['memoryPerCore'] = df['internalMemory'] / df['numCores'].replace(0, 1)
        df['memoryPerCore'].fillna(0, inplace=True)

        if df['memoryPerCore'].isna().any():
            logger.error("NaN detected in 'memoryPerCore'. Rows with NaN: \n%s", df[df['memoryPerCore'].isna()])
        
        # Battery Efficiency (battery power per screen area)
        df['batteryEfficiency'] = df['batteryPower'] / df['screenArea']
        df['batteryEfficiency'].fillna(0, inplace=True)

        if df['batteryEfficiency'].isna().any():
            logger.error("NaN detected in 'batteryEfficiency'. Rows with NaN: \n%s", df[df['batteryEfficiency'].isna()])

        # Final NaN Check for Entire DataFrame
        if df.isna().any().any():
            logger.error("NaN detected in DataFrame after feature engineering. Full rows with NaN: \n%s", df[df.isna().any(axis=1)])

        df.fillna(0, inplace=True)  # Replace any remaining NaN values with 0
        
        return df




    def predict_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send prediction request for a batch of devices, ensuring no NaN values exist."""
        try:
            # Remove 'id' from each device in the batch before sending to the API
            batch_for_prediction = [{key: value for key, value in device.items() if key != 'id'} for device in batch]

            # Check for NaN values before sending the batch
            for device in batch_for_prediction:
                if any(pd.isna(value) for value in device.values()):
                    logger.error("NaN value detected in batch data: %s", device)
                    raise ValueError("Batch contains NaN values, aborting request.")

            response = requests.post(
                f"{self.api_url}/devices/predict/bulk",
                json=batch_for_prediction,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()

            # Append 'id' back to each prediction result for easier identification
            predictions = response.json()
            for original_device, prediction in zip(batch, predictions):
                prediction['id'] = original_device['id']

            return predictions

        except requests.exceptions.RequestException as e:
            logger.error(f"Error predicting batch: {str(e)}")
            raise
        except ValueError as ve:
            logger.error(f"Validation Error in batch: {str(ve)}")
            raise


    def process_devices(self, devices_df: pd.DataFrame) -> None:
        """Process all devices in batches using threading."""
        # Compute engineered features
        devices_df = self.compute_engineered_features(devices_df)
        
        # Ensure features are in the correct order
        devices_df = devices_df[FEATURE_NAMES]
        
        batches = [devices_df[i:i + self.batch_size] 
                  for i in range(0, len(devices_df), self.batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for batch_df in batches:
                batch_data = [self.prepare_device_data(row) 
                            for _, row in batch_df.iterrows()]
                future = executor.submit(self.predict_batch, batch_data)
                futures.append(future)

            for future in tqdm(as_completed(futures), 
                             total=len(futures), 
                             desc="Processing batches"):
                try:
                    result = future.result()
                    self.results.extend(result)
                except Exception as e:
                    logger.error(f"Batch processing error: {str(e)}")
                    self.errors.append(str(e))

    def save_results(self, output_dir: str = "results") -> None:
        """Save prediction results and errors to files."""
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Save results
        if self.results:
            results_file = f"{output_dir}/predictions_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Results saved to {results_file}")
            
        # Save errors
        if self.errors:
            errors_file = f"{output_dir}/errors_{timestamp}.txt"
            with open(errors_file, 'w') as f:
                for error in self.errors:
                    f.write(f"{error}\n")
            logger.info(f"Errors saved to {errors_file}")

def main():
    try:
        # Load test data
        test_data = pd.read_csv('test.csv')

        # Pre-check for missing values before renaming columns
        if test_data.isna().any().any():
            logger.error("Missing values detected in initial test data. Rows with NaN: \n%s", test_data[test_data.isna().any(axis=1)])
            raise ValueError("Test data contains NaN values. Please clean the input data.")

        # Rename columns to match expected feature names
        column_mapping = {
            'id': 'id',
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
            'wifi': 'wifi'
        }
        test_data = test_data.rename(columns=column_mapping)

        # Initialize runner
        runner = BulkPredictionRunner()

        # Process devices
        logger.info("Starting bulk prediction processing...")
        start_time = time.time()

        runner.process_devices(test_data)

        # Save results
        runner.save_results()

        # Log summary
        execution_time = time.time() - start_time
        logger.info(f"Processing completed in {execution_time:.2f} seconds")
        logger.info(f"Total predictions: {len(runner.results)}")
        logger.info(f"Total errors: {len(runner.errors)}")

    except Exception as e:
        logger.error(f"Error in bulk prediction process: {str(e)}")
        raise


    

if __name__ == "__main__":
    main()
