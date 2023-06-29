from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

di = DataIngestion()
train_path, test_path = di.initiate_data_ingestion()

dt = DataTransformation()
train, test, preprocessor = dt.initiate_data_transformation(train_path, test_path)