# populate.py improvements
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabasePopulator:
    def __init__(self, db_url: str, csv_path: str):
        self.db_url = db_url
        self.csv_path = csv_path
        self.engine = None
        self.df = None

    def connect_to_db(self) -> None:
        """Establish database connection"""
        try:
            self.engine = create_engine(self.db_url)
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise

    def load_csv(self) -> None:
        """Load and validate CSV data"""
        try:
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"CSV file loaded successfully: {self.csv_path}")
            
            # Validate required columns
            required_columns = {
                'id', 'battery_power', 'blue', 'clock_speed', 'dual_sim',
                'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt',
                'n_cores', 'pc', 'px_height', 'px_width', 'ram',
                'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi'
            }
            
            missing_columns = required_columns - set(self.df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

        except FileNotFoundError:
            logger.error(f"CSV file not found: {self.csv_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise

    def preprocess_data(self) -> None:
        """Preprocess the data before insertion"""
        try:
            # Rename columns to match database schema
            column_mapping = {
                'id': 'id',
                'blue': 'bluetooth',
                'fc': 'front_camera',
                'four_g': 'four_g',
                'int_memory': 'internal_memory',
                'm_dep': 'mobile_depth',
                'mobile_wt': 'mobile_weight',
                'n_cores': 'num_cores',
                'pc': 'primary_camera',
                'px_height': 'pixel_height',
                'px_width': 'pixel_width',
                'sc_h': 'screen_height',
                'sc_w': 'screen_width'
            }
            self.df = self.df.rename(columns=column_mapping)

            # Convert boolean columns
            boolean_columns = ['bluetooth', 'dual_sim', 'four_g', 'three_g', 
                             'touch_screen', 'wifi']
            for column in boolean_columns:
                self.df[column] = self.df[column].astype(bool)

            # Add price_range column if not present
            if 'price_range' not in self.df.columns:
                self.df['price_range'] = None

            logger.info("Data preprocessing completed successfully")

        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def validate_data(self) -> None:
        """Validate data types and values"""
        try:
            # Validate numeric columns are positive
            numeric_columns = [
                'battery_power', 'clock_speed', 'internal_memory',
                'mobile_depth', 'mobile_weight', 'num_cores',
                'primary_camera', 'pixel_height', 'pixel_width',
                'ram', 'screen_height', 'screen_width', 'talk_time'
            ]
            
            for column in numeric_columns:
                if (self.df[column] < 0).any():
                    raise ValueError(f"Negative values found in column: {column}")

            logger.info("Data validation completed successfully")

        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            raise

    def populate_database(self) -> None:
        """Populate the database with the processed data"""
        try:
            # Check if any records already exist
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT COUNT(*) FROM devices"))
                existing_count = result.scalar()
                
                if existing_count > 0:
                    logger.warning(f"Database already contains {existing_count} records")
                    user_input = input("Do you want to proceed with insertion? (y/n): ")
                    if user_input.lower() != 'y':
                        logger.info("Database population cancelled by user")
                        return

            # Insert data
            self.df.to_sql('devices', self.engine, if_exists='append', index=False)
            logger.info(f"Successfully inserted {len(self.df)} records into database")

        except SQLAlchemyError as e:
            logger.error(f"Database error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error populating database: {str(e)}")
            raise

    def run(self) -> None:
        """Run the complete population process"""
        try:
            self.connect_to_db()
            self.load_csv()
            self.preprocess_data()
            self.validate_data()
            self.populate_database()
            logger.info("Database population completed successfully")
        except Exception as e:
            logger.error(f"Database population failed: {str(e)}")
            raise

if __name__ == "__main__":
    DB_URL = 'postgresql://postgres:12345@localhost:5432/device_pricing'
    CSV_PATH = 'test.csv'
    
    populator = DatabasePopulator(DB_URL, CSV_PATH)
    populator.run()