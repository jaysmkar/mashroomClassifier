import os 
import sys 
import pandas as pd 
from dataclasses import dataclass 
from src.exception import CustomException
from src.logger import logging 
from sklearn.model_selection import train_test_split 

@dataclass 
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts', 'train.csv')
    test_data_path : str = os.path.join('artifacts', 'test.csv')
    raw_data_path : str = os.path.join('artifacts', 'data.csv')

class DataIngestion: 
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Starting the data ingestion method.')
        try:
            logging.info('data reading started')
            data = pd.read_csv("data\mashroom.csv")
            logging.info("data reading completed")
            
            logging.info('creating the folders train_data.csv, test_data.csv and raw_data.csv')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True) #DOBUT
            
            logging.info("train test split initiated.")
            train_set, test_set = train_test_split(data, test_size=0.2, rando_state=42)
            logging.info('train test split completed successfully.')
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True) 
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('data ingestion completed successfully.')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
