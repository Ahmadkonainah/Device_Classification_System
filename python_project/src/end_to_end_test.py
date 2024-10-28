# end_to_end_test.py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import sns
from sqlalchemy import create_engine, text
import requests
import logging
from typing import Dict, List, Any
import json
from datetime import datetime
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EndToEndTester:
    def __init__(self, 
                 database_url: str,
                 api_base_url: str,
                 test_data_path: str = None,
                 max_workers: int = 5):
        self.database_url = database_url
        self.api_base_url = api_base_url
        self.test_data_path = test_data_path
        self.max_workers = max_workers
        self.engine = None
        self.test_results: Dict = {}
        
        # Create output directory for test results
        self.output_dir = Path('test_results')
        self.output_dir.mkdir(exist_ok=True)

    def connect_to_db(self) -> None:
        """Establish database connection"""
        try:
            self.engine = create_engine(self.database_url)
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise

    def load_test_data(self) -> pd.DataFrame:
        """Load test data from file or database"""
        try:
            if self.test_data_path:
                data = pd.read_csv(self.test_data_path)
                # Map CSV columns to API expected format
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
                    'wifi': 'wifi'
                }
                return data.rename(columns=column_mapping)
            else:
                with self.engine.connect() as connection:
                    query = text("SELECT * FROM devices WHERE price_range IS NULL")
                    return pd.read_sql(query, connection)
        except Exception as e:
            logger.error(f"Error loading test data: {str(e)}")
            raise
    def test_api_health(self) -> bool:
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.api_base_url}/health")
            if response.status_code == 200:
                logger.info("API health check succeeded")
                return True
            else:
                logger.error(f"API health check returned status code {response.status_code}")
                logger.error(f"Response content: {response.text}")
                return False
        except Exception as e:
            logger.error(f"API health check failed: {str(e)}")
            return False


    def predict_single_device(self, device: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single device"""
        try:
            device_id = device['id']
            # Remove id from the request payload
            prediction_data = {k: v for k, v in device.items() if k != 'id'}
            
            response = requests.post(
                f"{self.api_base_url}/predict",  # Changed from /devices/predict/{device_id}
                json=prediction_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = {
                    'id': device_id,
                    'prediction': response.json().get('priceRange'),
                    'status': 'success',
                    'response_time': response.elapsed.total_seconds()
                }
            else:
                result = {
                    'id': device_id,
                    'status': 'error',
                    'error': response.text,
                    'response_time': response.elapsed.total_seconds()
                }
            
            return result
                
        except Exception as e:
            logger.error(f"Prediction error for device {device.get('id')}: {str(e)}")
            return {
                'id': device.get('id'),
                'status': 'error',
                'error': str(e)
            }

    def validate_prediction(self, result: Dict[str, Any]) -> None:
        """Validate prediction against database"""
        try:
            if result['status'] != 'success':
                return
                
            device_id = result['id']
            predicted_price = result['prediction']
            
            with self.engine.connect() as connection:
                query = text("SELECT price_range FROM devices WHERE id = :id")
                db_price = connection.execute(query, {"id": device_id}).scalar()
                
                if db_price is not None:
                    result['validation'] = {
                        'actual_price': db_price,
                        'matches': db_price == predicted_price
                    }
                
        except Exception as e:
            logger.error(f"Validation error for device {result.get('id')}: {str(e)}")
            result['validation_error'] = str(e)

    def run_parallel_tests(self, devices: List[Dict[str, Any]]) -> None:
        """Run predictions in parallel"""
        try:
            # Limit to first 10 devices for testing
            devices = devices[:10]
            logger.info(f"Testing with {len(devices)} devices")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_device = {
                    executor.submit(self.predict_single_device, device): device
                    for device in devices
                }
                
                for future in as_completed(future_to_device):
                    device = future_to_device[future]
                    try:
                        result = future.result()
                        self.validate_prediction(result)
                        self.test_results[result['id']] = result
                    except Exception as e:
                        logger.error(f"Error processing device {device.get('id')}: {str(e)}")
            
            logger.info(f"Completed testing {len(self.test_results)} devices")
                        
        except Exception as e:
            logger.error(f"Error in parallel testing: {str(e)}")
            raise

    def generate_test_report(self) -> None:
        """Generate test results report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Calculate statistics
            total_tests = len(self.test_results)
            successful_tests = sum(1 for r in self.test_results.values() 
                                 if r['status'] == 'success')
            successful_validations = sum(1 for r in self.test_results.values()
                                       if r.get('validation', {}).get('matches', False))
            
            avg_response_time = np.mean([r['response_time'] 
                                       for r in self.test_results.values()
                                       if 'response_time' in r])
            
            report = {
                'timestamp': timestamp,
                'summary': {
                    'total_tests': total_tests,
                    'successful_tests': successful_tests,
                    'successful_validations': successful_validations,
                    'average_response_time': avg_response_time
                },
                'detailed_results': self.test_results
            }
            
            # Save report
            report_path = self.output_dir / f'test_report_{timestamp}.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
                
            logger.info(f"Test report generated: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating test report: {str(e)}")
            raise

    def run_tests(self) -> None:
        """Run complete test suite"""
        try:
            logger.info("Starting end-to-end tests")
            
            # Check API health
            if not self.test_api_health():
                raise RuntimeError("API health check failed")
                
            # Connect to database
            self.connect_to_db()

            # Load test data
            test_data = self.load_test_data()
            devices = test_data.to_dict('records')
            
            if not devices:
                raise ValueError("No test devices found")
                
            # Run tests in parallel
            start_time = time.time()
            self.run_parallel_tests(devices)
            end_time = time.time()
            
            # Generate report
            self.generate_test_report()
            
            # Log summary
            logger.info(f"Tests completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Total devices tested: {len(devices)}")
            logger.info(f"Successful predictions: {sum(1 for r in self.test_results.values() if r['status'] == 'success')}")
            
        except Exception as e:
            logger.error(f"Test suite failed: {str(e)}")
            raise

class TestResultAnalyzer:
    """Analyzes test results and generates visualizations"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.latest_results = None

    def load_latest_results(self) -> None:
        """Load the most recent test results"""
        try:
            result_files = list(self.results_dir.glob('test_report_*.json'))
            if not result_files:
                raise FileNotFoundError("No test result files found")
                
            latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
            with open(latest_file) as f:
                self.latest_results = json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading test results: {str(e)}")
            raise

    def generate_visualizations(self) -> None:
        """Generate visualizations of test results"""
        try:
            if not self.latest_results:
                self.load_latest_results()

            # Create visualization directory
            viz_dir = self.results_dir / 'visualizations'
            viz_dir.mkdir(exist_ok=True)

            # Response time distribution
            response_times = [
                r['response_time'] 
                for r in self.latest_results['detailed_results'].values()
                if 'response_time' in r
            ]
            
            plt.figure(figsize=(10, 6))
            plt.hist(response_times, bins=30)
            plt.title('API Response Time Distribution')
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Count')
            plt.savefig(viz_dir / 'response_times.png')
            plt.close()

            # Prediction accuracy by price range
            predictions = []
            actuals = []
            for result in self.latest_results['detailed_results'].values():
                if (result['status'] == 'success' and 
                    'validation' in result and 
                    'actual_price' in result['validation']):
                    predictions.append(result['prediction'])
                    actuals.append(result['validation']['actual_price'])

            if predictions and actuals:
                plt.figure(figsize=(8, 8))
                cm = confusion_matrix(actuals, predictions)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Prediction Confusion Matrix')
                plt.xlabel('Predicted Price Range')
                plt.ylabel('Actual Price Range')
                plt.savefig(viz_dir / 'confusion_matrix.png')
                plt.close()

            logger.info(f"Visualizations generated in {viz_dir}")

        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise

def setup_monitoring():
    """Setup monitoring and alerting"""
    try:
        # Configure logging handler for monitoring
        monitor_handler = logging.FileHandler('monitoring.log')
        monitor_handler.setLevel(logging.WARNING)
        monitor_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        monitor_handler.setFormatter(monitor_formatter)
        logger.addHandler(monitor_handler)
        
        # You could add additional monitoring setup here
        # For example, connecting to a monitoring service
        
    except Exception as e:
        logger.error(f"Error setting up monitoring: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration
    DATABASE_URL = 'postgresql://postgres:12345@localhost:5432/device_pricing'
    API_BASE_URL = 'http://localhost:5000'
    TEST_DATA_PATH = 'test.csv'
    MAX_WORKERS = 5  # Reduced from default

    try:
        setup_monitoring()
        tester = EndToEndTester(
            database_url=DATABASE_URL,
            api_base_url=API_BASE_URL,
            test_data_path=TEST_DATA_PATH,
            max_workers=MAX_WORKERS
        )
        tester.run_tests()
        logger.info("End-to-end testing completed successfully")
        
    except Exception as e:
        logger.error(f"End-to-end testing failed: {str(e)}")
        raise