from black_friday.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from black_friday.entity.config_entity import DataValidationConfig
from black_friday.exception import BlackFridayException
from black_friday.constant import *
from black_friday.util.util import read_yaml_file
import os, sys
from black_friday.logger import logging
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import pandas as pd
import numpy as np
import json

class DataValidation:
    def __init__(self,
                 data_ingestion_artifact : DataIngestionArtifact,
                 data_validation_config : DataValidationConfig):
        try:
            logging.info(f"{'>>' * 30} Data Validation log is Started {'<<' * 30}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise BlackFridayException(e, sys) from e
    
    def is_train_test_exists(self)->bool:
        try:
            logging.info("Checking the train and test data sets are available")
            train_file_path_exist = os.path.exists(self.data_ingestion_artifact.train_file_path)
            test_file_path_exist = os.path.exists(self.data_ingestion_artifact.test_file_path)
            if train_file_path_exist and test_file_path_exist:
                logging.info("Both train and test data sets are available")
                return True
            else:
                raise Exception("Train or Test data set is not available")
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def get_train_test(self):
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            return train_df, test_df
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def validate_schema(self):
        try:
            logging.info("Validating the train and test data sets and schema file.")
            train_df, test_df = self.get_train_test()
            schema = read_yaml_file(file_path=self.data_validation_config.schema_file_path)
            valiadte = False
            if list(train_df.columns) == list(test_df.columns):
                if len(train_df.columns) <= len(schema['columns']):
                    if list(train_df.columns) == list(test_df.columns) == list(schema['columns'].keys()):
                        for i in schema['domain_value'].keys():
                            train = set(train_df[i].unique()) 
                            test = set(test_df[i].unique()) 
                            sche = set(schema['domain_value'][i])
                            if train.issubset(sche) and test.issubset(sche):    
                                continue
                            else:
                                unknown_domain_values = [j for j in train if j not in sche]
                                logging.info(f'Unknown domain values in {i} : [{unknown_domain_values}]')
                                logging.info('We are going to remove these from the dataset')
                                index = []
                                for j in unknown_domain_values:
                                    index.append(list(train_df[train_df[i] == j].index))
                                index = list(np.array(index).reshape(-1))[0]
                                train_df.drop(index=index, inplace=True)
                                index = []
                                for j in unknown_domain_values:
                                    index.append(list(test_df[test_df[i] == j].index))
                                index = list(np.array(index).reshape(-1))[0]
                                test_df.drop(index=index, inplace=True)
                                continue
                        validate =  True
                        logging.info("Validation of train_df, test_df and schema file successfully completed.")
                        return validate, train_df, test_df
                    else:
                        raise Exception("Datasets don't have necessary columns")       
                else:
                    logging.info('Unnecessary columns in the dataset. So we are removing those columns')
                    columns = [i for i in train_df.columns if i not in schema['columns'].keys()]
                    train_df.drop(columns=columns, inplace=True)
                    columns = [i for i in test_df.columns if i not in schema['columns'].keys()]
                    test_df.drop(columns=columns, inplace=True)
                    logging.info('Sucessfuly removed the unnecessary columns from the train and test dataset')
                    if list(train_df.columns) == list(test_df.columns) == list(schema['columns'].keys()):
                        for i in schema['domain_value'].keys():
                            train = set(train_df[i].unique()) 
                            test = set(test_df[i].unique()) 
                            sche = set(schema['domain_value'][i])
                            if train.issubset(sche) and test.issubset(sche):
                                continue
                            else:
                                unknown_domain_values = [j for j in train if j not in sche]
                                logging.info(f'Unknown domain values in {i} : [{unknown_domain_values}]')
                                logging.info('We are going to remove these from the dataset')
                                index = []
                                for j in unknown_domain_values:
                                    index.append(list(train_df[train_df[i] == j].index))
                                index = list(np.array(index).reshape(-1))[0]
                                train_df.drop(index=index, inplace=True)
                                index = []
                                for j in unknown_domain_values:
                                    index.append(list(test_df[test_df[i] == j].index))
                                index = list(np.array(index).reshape(-1))[0]
                                test_df.drop(index=index, inplace=True)
                                continue
                        validate = True
                        logging.info("Validation of train_df, test_df and schema file successfully completed.")
                        return validate, train_df, test_df
                    else:
                        raise Exception("Datasets don't have necessary columns")                 
            else:
                raise Exception("Train and Test Data Sets don't have common columns")    
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def validated_train_test_file_path(self):
        try:
            _, train_df, test_df = self.validate_schema()
            validated_train_file_path = os.path.join(
                self.data_validation_config.validated_train_dir,
                "black_friday_train_data.csv"
            )
            validated_test_file_path = os.path.join(
                self.data_validation_config.validated_test_dir,
                "black_friday_test_data.csv"
            )
            
            if train_df is not None:
                os.makedirs(self.data_validation_config.validated_train_dir)
                logging.info(f"Validated data set file path : {validated_train_file_path}")
                train_df.to_csv(validated_train_file_path, index=False)
                
            if train_df is not None:
                os.makedirs(self.data_validation_config.validated_test_dir)
                logging.info(f"Validated data set file path : {validated_test_file_path}")
                test_df.to_csv(validated_test_file_path, index=False)
                
            return validated_train_file_path, validated_test_file_path
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def get_save_data_drift_report(self, train_file_path, test_file_path):
        try:
            profile = Profile(sections=[DataDriftProfileSection()])
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            profile.calculate(train_df, test_df)
            report = json.loads(profile.json())
            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)
            
            with open(report_file_path, "w") as file_path:
                json.dump(report, file_path, indent=6)
                
            return report
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def save_data_drift_page_report(self, train_file_path, test_file_path):
        try:
            dashboard = Dashboard(tabs=[DataDriftTab()])
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            dashboard.calculate(train_df, test_df)
            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir, exist_ok=True)
            dashboard.save(report_page_file_path)
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def is_data_drift_found(self, train_file_path, test_file_Path)-> bool:
        try:
            data_drift = set()
            report = self.get_save_data_drift_report(train_file_path=train_file_path,
                                                     test_file_path=test_file_Path)
            train_df = pd.read_csv(train_file_path)
            columns = train_df.columns
            for i in columns:
                data_drift.add(report['data_drift']['data']['metrics']['Gender']['drift_detected'])
            if data_drift=={False}:
                logging.info(f"Detection of data drift : {data_drift}")
                logging.info("Data is not drifted")
                return True
            else:
                logging.info(f"Detection of data drift : {data_drift}")
                message = "Data is Drifted"
                raise Exception(message)
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    @staticmethod
    def outliers(df, i):
        try:
            q1, q2, q3 = np.quantile(df[i], [0.25, 0.5, 0.75])
            iqr = q3 - q1
            upper_whisker = q3 + (3 * iqr)
            lower_whisker = q1 - (3 * iqr)
            percentage = (len(df[(df[i] > upper_whisker) | (df[i] < lower_whisker)])/len(df)) * 100
            return f'Outliers percentage of {i} : {percentage}'
        except Exception as e:
            raise Exception(e, sys) from e
    
        
    def exploratory_data_analysis(self, train_file_path) :
        try:
            train_file_path = train_file_path
            train_df = pd.read_csv(train_file_path)
            schema = read_yaml_file(self.data_validation_config.schema_file_path)
            target_variable = schema['target_column'][0]
            df = train_df.drop(target_variable, axis = 1)
            range = df.shape
            columns = df.columns
            null_value_count = sum(df.isnull().sum())
            numerical_columns = []
            categorical_columns = []
            outliers = []
            threshold = 30
            for i in columns:
                l = len(df[i].unique())
                if l > threshold:
                    numerical_columns.append(i)
                else:
                    categorical_columns.append(i)
            
            unique_values_of_target_variable = train_df[target_variable].unique()
            percentage = []
            for i in numerical_columns:
                outliers.append(DataValidation.outliers(df=df, i=i))
    
            logging.info(f'''Imbalanced dataset : {percentage}'
            Range : {range}\n
            columns : {columns}\n
            Null_Values : {null_value_count}\n
            numerical_columns : {numerical_columns}\n
            categorical_columns : {categorical_columns}
            outliers : {outliers}''')
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            self.is_train_test_exists()
            validated_train_file_path, validated_test_file_path = self.validated_train_test_file_path()
            self.get_save_data_drift_report(train_file_path=validated_train_file_path,
                                            test_file_path=validated_test_file_path)
            self.save_data_drift_page_report(train_file_path=validated_train_file_path,
                                             test_file_path=validated_test_file_path)
            self.is_data_drift_found(train_file_path=validated_train_file_path,
                                     test_file_Path=validated_test_file_path)
            self.exploratory_data_analysis(train_file_path=validated_train_file_path)
            data_validation_artifact = DataValidationArtifact(
                validated_train_file_path=validated_train_file_path,
                validated_test_file_path=validated_test_file_path,
                is_validated=True,
                schema_file_path=self.data_validation_config.schema_file_path,
                report_file_path=self.data_validation_config.report_file_path,
                report_page_file_path=self.data_validation_config.report_page_file_path,
                message="Data Validation Performed Sucessfully"
            )
            logging.info(f"Data Validation Artifact : {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def __del__(self):
            logging.info(f"{'>>' * 20} Data Validation log completed {'<<' * 20}\n\n")