from black_friday.logger import logging
from black_friday.exception import BlackFridayException
import os,sys
from black_friday.entity.artifact_entity import DataValidationArtifact, \
    ModelTrainerArtifact, ModelEvaluationArtifact
from black_friday.entity.config_entity import ModelEvaluationConfig
from black_friday.util.util import read_yaml_file, write_yaml_file, load_data
from black_friday.constant import BEST_MODEL_KEY, HISTORY_KEY, MODEL_PATH_KEY
from black_friday.util.util import load_object
import numpy as np
from black_friday.entity.model_factory import *


class ModelEvaluation:
    def __init__(self, data_validation_artifact : DataValidationArtifact,
                 model_trainer_artifact : ModelTrainerArtifact,
                 model_evaluation_config : ModelEvaluationConfig):
        try:
            logging.info(f"Model Evaluation log is Started")
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_config = model_evaluation_config
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def get_best_model(self):
        try:
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path
            if not os.path.exists(model_evaluation_file_path):
                write_yaml_file(model_evaluation_file_path)
                return model
            model_evaluation_content = read_yaml_file(model_evaluation_file_path)
            model_evaluation_content = dict() if model_evaluation_content is None else model_evaluation_content
            if BEST_MODEL_KEY not in model_evaluation_content:
                return model
            model = load_object(file_path=model_evaluation_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def update_evaluation_report(self, model_evaluation_artifact : ModelEvaluationArtifact):
        try:
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_evaluation_content = read_yaml_file(eval_file_path)
            model_evaluation_content = dict() if model_evaluation_content is None else model_evaluation_content
            
            previous_best_model = None
            
            if BEST_MODEL_KEY in model_evaluation_content:
                previous_best_model = model_evaluation_content[BEST_MODEL_KEY]
                
            eval_result = {
                BEST_MODEL_KEY:
                    {
                        MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path
                    }
            }
            
            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.timestamp : previous_best_model}
                if HISTORY_KEY not in model_evaluation_content:
                    model_evaluation_content.update({HISTORY_KEY : model_history})
                else:
                    model_evaluation_content[HISTORY_KEY].update(model_history)
                    
            model_evaluation_content.update(eval_result)
            
            logging.info(f"Updated eval result : {model_evaluation_content}")
            
            write_yaml_file(file_path=eval_file_path,
                            data = model_evaluation_content)
            
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path
            trained_model_obj = load_object(trained_model_file_path)
            
            train_file_path = self.data_validation_artifact.validated_train_file_path
            test_file_path = self.data_validation_artifact.validated_test_file_path
            
            schema_file_path = self.data_validation_artifact.schema_file_path
            schema = read_yaml_file(schema_file_path)
            
            logging.info('load the datasets of train and test')
            train_dataframe = load_data(file_path=train_file_path,
                                        schema_file_path=schema_file_path)
            test_dataframe = load_data(file_path=test_file_path,
                                       schema_file_path=schema_file_path)
            
            target_feature = schema['target_column'][0]
            
            logging.info("Converting the target variable into numpy array")
            train_target_arr = np.array(train_dataframe[target_feature])
            test_target_arr = np.array(test_dataframe[target_feature])
            
            logging.info(f"Drop the target variable from the test and train datasets")
            train_dataframe.drop(columns=[target_feature], axis=1, inplace=True)
            test_dataframe.drop(columns=[target_feature], axis=1, inplace=True)
            
            model = self.get_best_model()
            
            if model is None:
                logging.info('Not Found Existing Model. Hence the trained model')
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True,
                    evaluated_model_path=trained_model_file_path
                )
                self.update_evaluation_report(model_evaluation_artifact=model_evaluation_artifact)
                logging.info(f"Model accepted.Model eval artifact : {model_evaluation_artifact}")
                return model_evaluation_artifact
            
            model_list = [model, trained_model_obj]
            
            metric_info_artifact : MetricInfoArtifact = evaluate_regression_model(
                model_list=model_list,
                X_train=train_dataframe,
                y_train=train_target_arr,
                X_test = test_dataframe,
                y_test = test_target_arr,
                base_accuracy=self.model_trainer_artifact.model_accuracy
            )
            
            logging.info(f'Model Evaluation is accepted. Metric Info Artifact : {metric_info_artifact}')
            
            if metric_info_artifact is None:
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=False,
                    evaluated_model_path=trained_model_file_path
                )
                logging.info(model_evaluation_artifact)
                return model_evaluation_artifact
            
            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=True,
                    evaluated_model_path=trained_model_file_path
                )
                self.update_evaluation_report(model_evaluation_artifact=model_evaluation_artifact)
                logging.info(f'Model is accepted. Model evaluated artifact : {model_evaluation_artifact}')
            else:
                logging.info('No model better than the existing model. Hence not accepting the model.')
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=False, 
                    evaluated_model_path=trained_model_file_path
                )
            return model_evaluation_artifact
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def __del__(self)-> str:
        logging.info(f'{">>" * 20} Model Evaluation log is completed {"<<" * 20}')