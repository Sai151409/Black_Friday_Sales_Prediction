from black_friday.entity.artifact_entity import DataIngestionArtifact
from black_friday.entity.config_entity import DataIngestionConfig
import os,sys
import gdown
import logging
from black_friday.exception import BlackFridayException
import pandas as pd
from sklearn.model_selection import train_test_split
from zipfile import ZipFile

class DataIngestion:
    def __init__(self, data_ingestion_config : DataIngestionConfig):
        try:
            logging.info(f"{'>>' * 30} Data Ingestion is Started {'<<' * 30}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def download_black_friday_dataset(self):
        try:
            tgz_download_dir = self.data_ingestion_config.tgz_download_dir
            
            if os.path.exists(tgz_download_dir):
                os.remove(tgz_download_dir)
            os.makedirs(tgz_download_dir, exist_ok=True)
            
            logging.info(f"Download the dataset from google drive link [{self.data_ingestion_config.download_url}]\
                into dir : [{self.data_ingestion_config.tgz_download_dir}]")
            gdown.download(url=self.data_ingestion_config.download_url,
                           output=self.data_ingestion_config.tgz_download_dir,
                           quiet=False, fuzzy=True)
            logging.info("Downloaded the dataset Successfully")
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def extract_the_data_set(self):
        try:
            tgz_file_name = os.listdir(self.data_ingestion_config.tgz_download_dir)[0]
            tgz_file_path = os.path.join(self.data_ingestion_config.tgz_download_dir,
                                         tgz_file_name)
            raw_dir = self.data_ingestion_config.raw_data_dir
            if os.path.exists(raw_dir):
                os.remove(raw_dir)
            
            os.makedirs(raw_dir, exist_ok=True)
            logging.info(f"Extract the dataset from {tgz_file_path} to \
                dir {self.data_ingestion_config.raw_data_dir}")
            with ZipFile(tgz_file_path, "r") as file:
                file.extractall(raw_dir)
            logging.info("Extracted data Successfully")
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def split_data_as_train_test(self) -> DataIngestionArtifact:
        try:
            raw_data = self.data_ingestion_config.raw_data_dir
            file_name = os.listdir(raw_data)[0]
            file_path = os.path.join(raw_data, file_name)
            logging.info(f"Reading csv file : {file_path}")
            dataframe = pd.read_csv(file_path)
            dataframe.drop(columns=['User_ID', 'Product_ID'], inplace=True)
            
            
            start_train_set=None
            start_train_set=None
            
            logging.info("Splitting the dataset into train and tset")
            start_train_set, start_test_set = train_test_split(dataframe, test_size=0.3, random_state=42)
            
            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir, "black_friday_train_data.csv")
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir, "black_friday_test_data.csv")
            
            if start_train_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
                logging.info(f"Exporting data set into file : {[train_file_path]}")
                start_train_set.to_csv(train_file_path, index=False)
                
            if start_test_set is not None:
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)
                logging.info(f"Exporting data set into file : {[test_file_path]}")
                start_test_set.to_csv(test_file_path, index=False)
                
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=train_file_path, 
                test_file_path=test_file_path,
                is_ingested=True,
                message="Data Ingestion Completed Successfully."
            )
            logging.info(f"Data Ingestion Artifact : {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.download_black_friday_dataset()
            self.extract_the_data_set()
            return self.split_data_as_train_test()
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def __del__(self):
        logging.info(f"{'>>' * 30} Data Ingestion log Completed {'<<' * 30}\n\n")