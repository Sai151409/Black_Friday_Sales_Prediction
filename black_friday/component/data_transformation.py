from black_friday.exception import BlackFridayException
from black_friday.logger import logging
from black_friday.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from black_friday.entity.config_entity import DataTransformationConfig
import os, sys
from sklearn.base import BaseEstimator, TransformerMixin
from black_friday.constant import COLUMN_STAY_IN_CURRENT_CITY_YEARS, COLUMN_AGE, \
    SCHEMA_CATEGORICAL_COLUMN_KEY, SCHEMA_TARGET_VALUE_KEY
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder 
from black_friday.util.util import read_yaml_file, load_data, save_numpy_array, save_object
import numpy as np


class DataTransformation:
    def __init__(self, data_validation_artifact : DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            logging.info(f"{'>>' * 20} Data Transformation log is started {'<<' * 20}")
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def get_transformer_object(self) -> ColumnTransformer:
        try:
            ord_cols = ['Age', 'Stay_In_Current_City_Years']
            ohe_cols = ['Gender', 'City_Category']
            missing_columns = ['Product_Category_2', 'Product_Category_3']
            
            preprocessing = ColumnTransformer(
                [
                    ('onehot', OneHotEncoder(drop='first'), ohe_cols),
                    ('ord', OrdinalEncoder(), ord_cols),
                    ('impute', SimpleImputer(strategy='most_frequent'), missing_columns)
                    ], remainder='passthrough')
            
            return preprocessing
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Object preprocesisng object.")
            preprocessing_obj = self.get_transformer_object()
            
            logging.info("Obtaining the train and test file path")
            train_file_path = self.data_validation_artifact.validated_train_file_path
            test_file_path = self.data_validation_artifact.validated_test_file_path
            schema_path = self.data_validation_artifact.schema_file_path
            
            logging.info("Loading train and test dataset as pandas dataframe.")
            train_df = load_data(file_path=train_file_path,
                                 schema_file_path=schema_path)
            test_df = load_data(file_path=test_file_path,
                                schema_file_path=schema_path)
            schema = read_yaml_file(
                file_path=schema_path
            )
            
            target_column = schema[SCHEMA_TARGET_VALUE_KEY]
            logging.info("Splitting input and output dataframe from the dataframe")
            input_train_df = train_df.drop(columns=[target_column][0], axis = 1)
            output_train_df = train_df[target_column]
            
            input_test_df = test_df.drop(columns=[target_column][0], axis = 1)
            output_test_df = test_df[target_column]
            
            logging.info("Applying the preprocessing object to the train and test dataframe")
            
            input_train_arr = preprocessing_obj.fit_transform(input_train_df)
            input_test_arr = preprocessing_obj.transform(input_test_df)

            
            train_arr = np.c_[input_train_arr, output_train_df]
            test_arr = np.c_[input_test_arr, output_test_df]
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir
            
            transformed_train_file_name = os.path.basename(train_file_path).replace(".csv", ".npz")
            transformed_test_file_name = os.path.basename(test_file_path).replace(".csv", ".npz")
            
            logging.info("Saving the transformed train and test numpy arrays")
            transformed_train_file_path = os.path.join(transformed_train_dir, transformed_train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, transformed_test_file_name)
            
            save_numpy_array(transformed_train_file_path, train_arr)
            save_numpy_array(transformed_test_file_path, test_arr)
            
            logging.info("Saving the preprocessing object")
            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_file_path
            
            save_object(preprocessing_obj_file_path, preprocessing_obj)
            
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=transformed_train_file_path,
                transformed_test_file_path=transformed_test_file_path,
                is_transformed=True,
                preprocessed_object_file_path=preprocessing_obj_file_path,
                message="Data Transfromation Performed Successfully"
            )
            
            logging.info(f"Data Transformation Artifact : {data_transformation_artifact}")
            
            return data_transformation_artifact
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    
    def __del__(self):
        logging.info(f"{'>>' * 20} Data Transformation log is Completed {'<<' * 20}")
        
