import os, sys 
import pandas as pd 
import numpy as np 
from dataclasses import dataclass 
from src.logger import logging 
from src.exception import CustomException
from sklearn.model_selection import train_test_split

@dataclass 
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts/data_ingestion', 'train_data.csv')
    test_data_path = os.path.join('artifacts/data_ingestion', 'test_data.csv')
    raw_data_path = os.path.join('artifacts/data_ingestion', 'raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info('inside the initiate_data_ingestion method.')
        try: 
            logging.info('reading the data from csv file.')
            data = pd.read_csv('data\mashroom.csv')
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.data_ingestion_config.raw_data_path, index=False)

            logging.info('splitting the data into the training and testing files')
            train_set, test_set = train_test_split(data, test_size=0.30, random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data Ingestion completed successfully.")
            
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_set, test_set = obj.initiate_data_ingestion()
