import os
from datetime import datetime

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

CURRENT_TIME_STAMP = get_current_time_stamp()

COLUMN_STAY_IN_CURRENT_CITY_YEARS = 'Stay_In_Current_City_Years'
COLUMN_AGE = 'Age'

CONFIG_FILE_NAME = 'config.yaml'
CONFIG_DIR = 'config'
ROOT_DIR = os.getcwd()

CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, CONFIG_FILE_NAME)

SCHEMA_COLUMN_KEY = "columns"
SCHEMA_CATEGORICAL_COLUMN_KEY = "categorical_columns"
SCHEMA_TARGET_VALUE_KEY = "target_column"
SCHEM_DOMAIN_VALUE_KEY = "domain_value"


TRAINING_PIPELINE_CONFIG_KEY = "training_pipeline"
TRAINING_PIPELINE_NAME_KEY = 'pipeline'
TRAINING_PIPELINE_ARTIFACT_KEY = 'artifact'

DATA_INGESTION_CONFIG_KEY = 'data_ingestion'
DATA_INGESTION_DIR_KEY = 'data_ingestion_dir'
DATA_INGESTION_DONWLOAD_URL = 'download_url'
DATA_INGESTION_TGZ_DOWMNLOAD_DIR_KEY = "tgz_download_dir"
DATA_INGESTION_RAW_DATA_DIR_KEY = 'raw_data_dir'
DATA_INGESTION_INGESTED_DATA_DIR_KEY = 'ingested_data_dir'
DATA_INGESTION_INGESTED_TRAIN_DIR_KEY = 'ingested_train_dir'
DATA_INGESTION_INGESTED_TEST_DIR_KEY = 'ingested_test_dir'

DATA_VALIDATION_CONFIG_KEY = "data_validation"
DATA_VALIDATION_DIR_KEY = "data_validation_dir"
DATA_VALIDATION_VALIDATED_DATA_DIR_KEY = "validated_data_dir"
DATA_VALIDATION_VALIDATED_TRAIN_DIR_KEY = "validated_train_dir"
DATA_VALIDATION_VALIDATED_TEST_DIR_KEY = "validated_test_dir"
DATA_VALIDATION_SCHEMA_DIR_KEY = "schema_dir"
DATA_VALIDATION_SCHEMA_FILE_NAME_KEY = "schema_file_name"
DATA_VALIDATION_REPORT_FILE_NAME_KEY = 'report_file_name'
DATA_VALIDATION_REPORT_PAGE_FILE_NAME_KEY = 'report_page_file_name'

DATA_TRANSFORMATION_CONFIG_KEY = "data_transformation"
DATA_TRANSFORMATION_DIR_KEY = "data_transformation_dir"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR = "transformed_data_dir"
DATA_TRANSFORMATION_TRANSFORMED_TRAIN_DIR = "transformed_train_dir"
DATA_TRANSFORMATION_TRANSFORMED_TEST_DIR = "transformed_test_dir"
DATA_TRANSFORMATION_PREPROCESSED_DIR = "preprocessed_dir"
DATA_TRANSFORMATION_PREPROCESSED_FILE_NAME_KEY = "preprocessed_file_name"

MODEL_TRAINER_CONFIG_KEY = "model_trainer"
MODEL_TRAINER_DIR_KEY = "model_trainer_dir"
MODEL_TRAINER_TRAINED_MODEL_DIR = "trained_model_dir"
MODEL_TRAINER_TRAINED_MODEL_FILE_NAME = "trained_model_file_name"
MODEL_TRAINER_BASE_ACCURACY = "base_accuracy"
MODEL_TRAINER_MODEL_CONFIG_DIR = "model_config_dir"
MODEL_TRAINER_MODEL_CONFIG_FILE_NAME = "model_config_file_name"

MODEL_EVALUATION_CONFIG_KEY = "model_evaluation"
MODEL_EVALUATION_DIR_KEY = "model_evaluation_dir"
MODEL_EVALUATION_FILE_NAME_KEY = "model_evaluation_file_name"

MODEL_PUSHER_CONFIG_KEY = "model_pusher"
MODEL_PUSHER_EXPORT_DIR_KEY = "model_export_dir"

BEST_MODEL_KEY = 'best_model'
HISTORY_KEY = 'history'
MODEL_PATH_KEY = 'model_path'

EXPERIMENT_DIR_KEY = "experiment"
EXPERIMENT_FILE_NAME = "experiment.csv"

TEST_DATA_KEY = 'test_data_set'
TEST_DATA_LINK = 'link'