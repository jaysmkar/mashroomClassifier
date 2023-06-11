import os 
import sys 
import numpy as np 
import pandas as pd 
from src.utils import save_object
from dataclasses import dataclass 
from src.exception import CustomException
from src.logger import logging 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from autoimpute.imputations import SingleImputer,MultipleImputer
from autoimpute.imputations.series import MultinomialLogisticImputer


@dataclass 
class DataTransformationConfig: 
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation: 
    def __init__(self):
        self.data_transformation_config = DataTransformation()
        
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation
        """
        try: 
            
            """
            There are only categorical columns in the dataset so performing the missing value imputations
            on the same categorical dataset.
            """
            cat_columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises%3F', 'odor',
                'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                'stalk-surface-below-ring', 'stalk-color-above-ring',
                'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type',
                'spore-print-color', 'population', 'habitat', 'class']
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SingleImputer(statergy="categorical")),
                    ("scalar", StandardScaler())
                ]
            )
            
            logging.info(f'Categorical columns are:{cat_columns}')

            preprocessor = ColumnTransformer(
                [
                    ("categorical_pipeline", cat_pipeline, cat_columns)
                ]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def data_transformation(self, train_path, test_path):
        
        try: 
            # 1. Instead of np.nan we have '?' values in stalk-root column we have to replace it by np.nan. __> done
            # 2. There is only one unique value in column veil-type so we can drop it then and there itself. --> done
            # 3. Using the single imputer to impute the np.nan values in the dataset. 
            # 4. In the target column there is 'p' and 'e' replace all the 'p':0 and 'e':1. --> done 
            # 5. Encoding rest all columns rather than target using get_dummies() and storing it in new_data variable.

            logging.info('train and test data files reading initiated.')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('train and test data reading files completed.')
            
            logging.info('obtaining the preprocessor object.')
            preprocessor_obj = self.get_data_transformer_object()
            logging.info('captured the preprocessor object information.')

            target_column = 'class'
            to_drop_column = 'veil-type'
            categorical_columns = ['cap-shape', 'cap-surface', 'cap-color', 'bruises%3F', 'odor',
                'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                'stalk-surface-below-ring', 'stalk-color-above-ring',
                'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type',
                'spore-print-color', 'population', 'habitat', 'class']
            
            #replacing the '?' values with the np.nan values.
            train_df["stalk-root"] =train_df["stalk-root"].replace('?',np.nan) 
            test_df["stalk-root"] = test_df["stalk-root"].replace('?', np.nan)
            
            # replacing class column in train_df and test_df p:0 and e:1
            logging.info("Replacing 'e' and 'p' started.")
            train_df['class'] = train_df['class'].replace({"'p'":0, "'e'":1})
            test_df['class'] = test_df['calss'].replace({"'p'": 0, "'e'": 1})
            logging.info("Replacing 'e' and 'p' ended.")
            
            
            #dividing the data into the input data and output data
            logging.info('Dividing the input and output information that is the input and target variables')
            input_feature_train_df = train_df.drop(columns=[target_column, to_drop_column], axis=1)
            target_feature_train_df = train_df['class']
            
            input_feature_test_df = test_df.drop(columns=[target_column, to_drop_column], axis=1)
            target_feature_test_df = test_df['class']
            logging.info("Dividing the information completed.")
            
            # encoding the data values using get_dummies() 
            logging.info("Encoding the train and test data using get_dummies() initiated.")
            column_list = ['cap-shape', 'cap-surface', 'cap-color', 'bruises%3F', 'odor',
                    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                    'stalk-surface-below-ring', 'stalk-color-above-ring',
                    'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type',
                    'spore-print-color', 'population', 'habitat']
            for column in column_list:
                train_df = pd.get_dummies(train_df, columns=[column], drop_first=True)
                test_df = pd.get_dummies(test_df, columns=[column], drop_first=True)
            logging.info("Encoding the train and test data using get_dummies() succeeded.")
            
            # applying the preprocessing object on data.
            logging.info("Application of preprocessor initiated.")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            logging.info("Aplication of the preprocessor succeeded.")
            
            # final conversion into the training and testing array. 
            logging.info('Final conversion of train and test data initiated.')
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info('Final conversion of train and test data succeded.')
            
            #finally saving the object. 
            logging.info('Saving object initiated.')
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            logging.info('Saving object completed.')
            
            return(
                train_arr,
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path
            )
  
        except Exception as e:
            raise CustomException(e,sys)