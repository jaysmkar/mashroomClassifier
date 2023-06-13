import os, sys 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from src.exception import CustomException
from src.logger import logging 
from dataclasses import dataclass 

from src.components.data_transformation import DataTransformationConfig, DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts/data_ingestion', 'train_data.csv')
    test_data_path = os.path.join('artifacts/data_ingestion', 'test_data.csv')
    raw_data_path = os.path.join('artifacts/data_ingestion', 'data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        
        try: 
        
            logging.info('reading the dataframe in data variable')
            data = pd.read_csv('data\mashroom.csv')
            
            logging.info('making the new directory in the artifacts folder.')
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('splitting into the training and testing data')
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            logging.info('Converting the train and the test files to the .csv format')
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('returning the training and test data paths')
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path  
            )
        except Exception as e:
            logging.info('Error occured in data ingestion stage')
            raise CustomException(e,sys)                                  


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path) 


