"""  
EDA Summary: 
1. Instead of np.nan we have '?' values in stalk-root column we have to replace it by np.nan. --> done
2. There is only one unique value in column veil-type so we can drop it then and there itself. --> done
3. Using the single imputer to impute the np.nan values in the dataset. --> object ready 
4. In the target column there is 'p' and 'e' replace all the 'p':0 and 'e':1 using .map()
5. Encoding rest all columns rather than target using get_dummies() and storing it in new_data variable.
"""

import os, sys 
import numpy as np 
import pandas as pd 
from src.logger import logging 
from src.exception import CustomException
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils import save_object 

@dataclass
class DataTransformationConfig():
    preprocessor_file_path = os.path.join('artifacts/data_transformation', 'preprocessor.pkl')

class DataTransformation():

    def __inti__(self):
        self.data_transformation_config = DataTransformationConfig() 
        
    def get_data_transformation_obj(self):
        """ 
        This function is only used for imputing the missing values in the dataset.
        """
        try:
            cat_pipeline = ['cap-shape', 'cap-surface', 'cap-color', 'bruises%3F', 'odor',
                            'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                            'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                            'stalk-surface-below-ring', 'stalk-color-above-ring',
                            'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type',
                            'spore-print-color', 'population', 'habitat']         
            categorical_pipeline = Pipeline(
                steps = [(
                    ("imputer", SimpleImputer(strategy='mode'))
                )]
            )
            # APPLYING THE COLUMNTRANSFORMER 
            preprocessor = ColumnTransformer([
                ('cat_pipeline', categorical_pipeline, cat_pipeline)
            ])
            # RETURNING THE PREPROCESSOR OBJECT 
            return preprocessor  
            
        except Exception as e:
            raise CustomException(e, sys)
        
    def remove_outlier_IQR(self, df, col):
        """ 
        This function is used to remove outlier from the numerical columns.
        """
        try: 
            logging.info('inside the remove_outlier_IQR class')
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3-Q1
            UL = Q3+1.5*IQR 
            LL = Q1-1.5*IQR
            df.loc[(df[col]> UL), col] = UL 
            df.loc[(df[col]< UL), col] = LL 
            return df 
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            logging.info('Inside the data initiate data transformation method.')
            
            # READING THE TRAIN AND TEST DATASETS 
            logging.info('Reading the train and test data path started.')
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            logging.info('Reading train and test data path completed.')

            # DELETING THE USELESS COLUMNS 
            logging.info('Deleting useless columns started.')
            train_data = train_data.drop(columns=['veil-type'], axis=1)
            test_data = test_data.drop(columns=['veil-type'], axis=1)
            logging.info('Deleting useless columns finished.')
            
            # REPLACING THE '?' BY NP.NAN VALUES 
            logging.info("Replacing '?' by np.nan values initialized")
            train_data = train_data['stalk-root'].replace('?', np.nan)
            test_data = test_data['stalk-root'].replace('?', np.nan)
            logging.info("Replacing '?' by np.nan values finished.")

            
            
            target_columns = "class"
            drop_columns = [target_columns]

            # SPLITTING THE DATA INTO TRAIN AND TEST DATA 
            logging.info('data splitting in dependent and independent columns initiated.')
            input_feature_train_data = train_data.drop(columns=drop_columns, axis=1)
            target_feature_train_data = train_data[target_columns]
            input_feature_test_data = test_data.drop(columns=drop_columns, axis=1)
            target_feature_test_data = test_data[target_columns]
            logging.info('data splitting in dependent and independent columns finished.')
            
            # APPLYING THE PREPROCESSOR OBJECT ON TRAIN AND TEST ARRAY. 
            logging.info('application of preprocessor object started.')
            preprocessor_obj = self.get_data_transformation_obj()
            input_train_array = preprocessor_obj.fit_transform(input_feature_train_data)
            input_test_array = preprocessor_obj.fit(input_feature_test_data)
            logging.info('application of preprocessor object finished.')
            
            train_array = np.c_[input_train_array, np.array(target_feature_train_data)]
            test_array = np.c_[input_test_array, np.array(target_feature_test_data)]

            
            ###################################################
            ##########Get dummies() method remaining########### 
            ###################################################
            
            save_object(file_path=self.data_transformation_config.preprocessor_file_path,
                        object_name=preprocessor_obj)
            
            return(
                train_array, 
                test_array,
                self.data_transformation_config.preprocessor_file_path
            )
            
        except Exception as e: 
            raise CustomException(e, sys)
        
        