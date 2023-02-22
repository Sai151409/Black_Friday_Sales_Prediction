import os, sys
from black_friday.constant import *
from black_friday.util.util import read_yaml_file
from black_friday.exception import BlackFridayException
from black_friday.logger import logging
from black_friday.entity.config_entity import DataIngestionConfig, DataValidationConfig,\
    DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig,\
    TrainingPipelineConfig

class Configuration:
    def __init__(self, config_file_path : str = CONFIG_FILE_PATH, 
                  time_stamp : str = CURRENT_TIME_STAMP):
        try:
            self.config = read_yaml_file(config_file_path)
            self.time_stamp = time_stamp
            self.training_pipeline_artifact = self.get_pipeline_artifact()
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            data_ingestion_info = self.config[DATA_INGESTION_CONFIG_KEY]
            data_ingestion_artifact = os.path.join(self.training_pipeline_artifact.artifact_dir, 
                                                   data_ingestion_info[DATA_INGESTION_DIR_KEY],
                                                   self.time_stamp)
            download_url = data_ingestion_info[DATA_INGESTION_DONWLOAD_URL]
            tgz_download_dir = os.path.join(data_ingestion_artifact,
                                            data_ingestion_info[DATA_INGESTION_TGZ_DOWMNLOAD_DIR_KEY])
            raw_data_dir = os.path.join(data_ingestion_artifact, 
                                        data_ingestion_info[DATA_INGESTION_RAW_DATA_DIR_KEY])
            ingested_train_dir = os.path.join(data_ingestion_artifact,
                                              data_ingestion_info[DATA_INGESTION_INGESTED_DATA_DIR_KEY],
                                              data_ingestion_info[DATA_INGESTION_INGESTED_TRAIN_DIR_KEY])
            ingested_test_dir = os.path.join(data_ingestion_artifact,
                                             data_ingestion_info[DATA_INGESTION_INGESTED_DATA_DIR_KEY],
                                             data_ingestion_info[DATA_INGESTION_INGESTED_TEST_DIR_KEY])
            data_ingestion_config = DataIngestionConfig(
                download_url=download_url,
                tgz_download_dir=tgz_download_dir,
                raw_data_dir=raw_data_dir,
                ingested_train_dir=ingested_train_dir,
                ingested_test_dir=ingested_test_dir
            )
            logging.info(f'Data Ingestion Config : {data_ingestion_config}')
            return data_ingestion_config
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            data_validation_info = self.config[DATA_VALIDATION_CONFIG_KEY]
            data_validation_artifact = os.path.join(
                self.training_pipeline_artifact.artifact_dir,
                data_validation_info[DATA_VALIDATION_DIR_KEY],
                self.time_stamp
            )
            validated_train_dir = os.path.join(
                data_validation_artifact,
                data_validation_info[DATA_VALIDATION_VALIDATED_DATA_DIR_KEY],
                data_validation_info[DATA_VALIDATION_VALIDATED_TRAIN_DIR_KEY]
            )
            validated_test_dir = os.path.join(
                data_validation_artifact,
                data_validation_info[DATA_VALIDATION_VALIDATED_DATA_DIR_KEY],
                data_validation_info[DATA_VALIDATION_VALIDATED_TEST_DIR_KEY]
            )
            schema_file_path = os.path.join(ROOT_DIR,
                                            data_validation_info[DATA_VALIDATION_SCHEMA_DIR_KEY],
                                            data_validation_info[DATA_VALIDATION_SCHEMA_FILE_NAME_KEY])
            report_file_path = os.path.join(data_validation_artifact,
                                            data_validation_info[DATA_VALIDATION_REPORT_FILE_NAME_KEY])
            report_page_file_path = os.path.join(
                data_validation_artifact,
                data_validation_info[DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY]
            )
            
            data_validation_config = DataValidationConfig(
                validated_train_dir=validated_train_dir,
                validated_test_dir=validated_test_dir,
                schema_file_path=schema_file_path,
                report_file_path=report_file_path,
                report_page_file_path=report_page_file_path,
                
            )
            
            logging.info(f"Data Validation Config : {data_validation_config}")
            return data_validation_config
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def get_data_transformation_config(self) -> DataTransformationConfig:
        try:
            data_transformation_info = self.config[DATA_TRANSFORMATION_CONFIG_KEY]
            data_transformation_artifact = os.path.join(
                self.training_pipeline_artifact.artifact_dir,
                data_transformation_info[DATA_TRANSFORMATION_DIR_KEY],
                self.time_stamp
            )
            
            transformed_train_dir = os.path.join(
                data_transformation_artifact,
                data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR],
                data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR]
            )
            transformed_test_dir = os.path.join(
                data_transformation_artifact,
                data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR],
                data_transformation_info[DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR]
            )
            preprocessed_file_path = os.path.join(
                data_transformation_artifact,
                data_transformation_info[DATA_TRANSFORMATION_PREPROCESSED_DIR],
                data_transformation_info[DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY]
            )
            
            data_transformation_config = DataTransformationConfig(
                transformed_train_dir=transformed_train_dir,
                transformed_test_dir=transformed_test_dir,
                preprocessed_file_path=preprocessed_file_path
            )
            
            logging.info(f"Data Transformation : {data_transformation_config}")
            return data_transformation_config
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        try:
            model_trainer_info = self.config[MODEL_TRAINER_CONFIG_KEY]
            model_trainer_artifact = os.path.join(
                self.training_pipeline_artifact.artifact_dir,
                model_trainer_info[MODEL_TRAINER_DIR_KEY],
                self.time_stamp
            )
            trained_model_file_path = os.path.join(
                model_trainer_artifact,
                model_trainer_info[MODEL_TRAINER_TRAINED_MODEL_DIR],
                model_trainer_info[MODEL_TRAINER_TRAINED_MODEL_FILE_NAME]
            )
            base_accuracy = model_trainer_info[MODEL_TRAINER_BASE_ACCURACY]
            model_config_file_path = os.path.join(
                ROOT_DIR,
                model_trainer_info[MODEL_TRAINER_MODEL_CONFIG_DIR],
                model_trainer_info[MODEL_TRAINER_MODEL_CONFIG_FILE_NAME]
            )
            model_trainer_config = ModelTrainerConfig(
                trained_model_file_path=trained_model_file_path,
                base_accuracy=base_accuracy,
                model_config_file_path=model_config_file_path
            )
            
            logging.info(f"Model Trainer Config : {model_trainer_config}")
            return model_trainer_config
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def get_model_evaluation_config(self)-> ModelEvaluationConfig:
        try:
            model_evaluation_info = self.config[MODEL_EVALUATION_CONFIG_KEY]
            model_evaluation_artifact = os.path.join(
                self.training_pipeline_artifact.artifact_dir,
                model_evaluation_info[MODEL_EVALUATION_DIR_KEY]
            )
            model_evaluation_file_path = os.path.join(
                model_evaluation_artifact,
                model_evaluation_info[MODEL_EVALUATION_FILE_NAME_KEY]
            )
            timestamp = self.time_stamp
            model_evaluation_config = ModelEvaluationConfig(
                model_evaluation_file_path=model_evaluation_file_path,
                timestamp=timestamp
            )
            logging.info(f"Model Evaluation Config : {model_evaluation_config}")
            return model_evaluation_config
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def get_model_pusher_config(self)-> ModelPusherConfig:
        try:
            time_stamp = f'{datetime.now().strftime("%Y%m%d%H%M%S")}'
            
            model_pusher_info = self.config[MODEL_PUSHER_CONFIG_KEY]
            export_dir_path = os.path.join(
                ROOT_DIR,
                model_pusher_info[MODEL_PUSHER_EXPORT_DIR_KEY], time_stamp
            )
            model_pusher_config = ModelPusherConfig(
                export_dir_path=export_dir_path
            )
            logging.info(f"Model Pusher : {model_pusher_config}")
            return model_pusher_config
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def get_pipeline_artifact(self) -> TrainingPipelineConfig:
        try:
            training_pipeline_config_info = self.config[TRAINING_PIPELINE_CONFIG_KEY]
            artifact_dir = os.path.join(
                ROOT_DIR,
                training_pipeline_config_info[TRAINING_PIPELINE_NAME_KEY],
                training_pipeline_config_info[TRAINING_PIPELINE_ARTIFACT_KEY]
            )
            training_pipeline_artifact = TrainingPipelineConfig(
                artifact_dir=artifact_dir
            )
            logging.info(f"Training Pipeline Artifact : {training_pipeline_artifact}")
            return training_pipeline_artifact
        except Exception as e:
            raise BlackFridayException(e, sys) from e