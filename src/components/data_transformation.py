import os, sys 
import pandas as pd 
import numpy as np
from src.logger import logging 
from src.exception import CustomException
from dataclasses import dataclass 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from src.utils import save_object
from sklearn.pipeline import Pipeline 


@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts/data_transformation', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() 
        
    def drop_veil_type(self, train_data, test_data):
        try:
            logging.info("Inside drop_veil_type method")
            column_to_drop = 'veil-type'
            train_data = train_data.drop(columns=[column_to_drop], axis=1)
            test_data = test_data.drop(columns=[column_to_drop], axis=1)
            
            return(
                train_data, 
                test_data
            )
        except Exception as e:
            raise CustomException(e, sys)
        
    def replace_p_e(self, train_data, test_data):
        try:
            logging.info("Inside replace_p_e method")
            train_data['class'] = train_data['class'].replace({"'p'":0, "'e'":1})
            test_data['class'] = test_data['class'].replace({"'p'":0, "'e'":1})
            return(
                train_data,
                test_data 
            )
        except Exception as e:
            raise CustomException(e, sys)
        
    def replace_question_mark(self, train_data, test_data):
        try:
            logging.info("Inside replace_question_mark method")
            train_data['stalk-root'] = train_data['stalk-root'].replace('?', np.nan)
            test_data['stalk-root'] = test_data['stalk-root'].replace('?', np.nan)
            return(
                train_data,
                test_data,
            )
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_data_transformation_obj(self):
        try:
            logging.info("Inside get_data_transformation_obj method")
            categorical_features = ['stalk-root']
            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = 'most_frequent'))
                ]
            )
            preprocessor = ColumnTransformer([
                ("cat_pipeline", categorical_pipeline, categorical_features)
            ])
            logging.info('get_data_transformation completed successfully.')
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

        
    def dummy_application(self, train_data, test_data):
        try:
            logging.info('Inside dummy_application method.')
            column_list = ['cap-shape', 'cap-surface', 'cap-color', 'bruises%3F', 'odor',
                            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                            'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                            'stalk-surface-below-ring', 'stalk-color-above-ring',
                            'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type',
                            'spore-print-color', 'population', 'habitat']
            for column in column_list:
                train_data = pd.get_dummies(train_data, columns=[column], drop_first=True)
                test_data = pd.get_dummies(test_data, columns=[column], drop_first=True)
            return(
                train_data,
                test_data
            )
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Inside initiate_data_transformation method.")
        try:
            logging.info('reading the training and testing data initiated.')
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            logging.info('reading the training and testing data completed.')
            
            logging.info('dropping the veil-type initiated.')
            train_data, test_data = self.drop_veil_type(train_data, test_data)
            logging.info('dropping the veil-type completed.')
            
            logging.info('replacing the p and e initiated.')
            train_data, test_data = self.replace_p_e(train_data, test_data)
            logging.info('replacing the p and e completed.')
            
            logging.info('replacing the question mark with np.nan initiated.')
            train_data, test_data = self.replace_question_mark(train_data, test_data)
            logging.info('replacing the question mark with np.nan completed.')
            
            preprocessor_object = self.get_data_transformation_obj()

            logging.info('seperating the training and testing data.')
            target_column_name = 'class'
            input_feature_train_data = train_data.drop(columns=[target_column_name], axis=1)
            target_feature_train_data = train_data[target_column_name]
            input_feature_test_data = test_data.drop(columns=[target_column_name], axis=1)
            target_feature_test_data = test_data[target_column_name]
            
            input_train_arr = preprocessor_object.fit_transform(input_feature_train_data)
            input_test_arr = preprocessor_object.transform(input_feature_test_data)

            train_array = np.c_[input_train_arr, np.array(target_feature_train_data)]
            test_array = np.c_[input_test_arr, np.array(target_feature_test_data)]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_object
            )
            
            logging.info('The data transformation completed successfully.')
            return(
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)    