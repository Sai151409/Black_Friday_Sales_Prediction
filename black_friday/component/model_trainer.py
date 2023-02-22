from black_friday.logger import logging
from black_friday.exception import BlackFridayException
import os, sys
from black_friday.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from black_friday.entity.config_entity import ModelTrainerConfig
from black_friday.util.util import load_numpy_array, load_object, save_object
from black_friday.entity.model_factory import ModelFactory, evaluate_regression_model,\
    GridSearchBestModel, MetricInfoArtifact
from typing import List

class BlackFridayEstimatorModel:
    def __init__(self, preprocessing_obj, trained_model_obj):
        try:
            self.preprocessing_obj = preprocessing_obj
            self.trained_model_obj = trained_model_obj
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def predict(self, X):
        try:
            transformed_obj = self.preprocessing_obj.transform(X)
            
            return self.trained_model_obj.predict(transformed_obj)
        except Exception as e:
            raise BlackFridayException(e, sys) from e
    
    def __repr__(self) -> str:
        return f"{type(self.trained_model_obj).__name__}()"
    
    def __str__(self) -> str:
        return f"{type(self.trained_model_obj).__name__}()"


class ModelTrainer:
    def __init__(self, 
                 data_transformation_artifact : DataTransformationArtifact,
                 model_trainer_config : ModelTrainerConfig):
        try:
            logging.info(f"{'>>' * 20} Model Trainer log is Started {'<<' *20}")
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def initiate_model_trainer(self):
        try:
            logging.info('load the files of transformed train array and test array')
            train_arr = load_numpy_array(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array(self.data_transformation_artifact.transformed_test_file_path)
            
            logging.info('Splitting the target and input features')
            X_train, y_train, X_test, y_test = train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1]
            
            logging.info('Intializing the model factory by giving the model config file path')
            model_factory = ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            
            logging.info(f"Excepted Base Accuracy : {self.model_trainer_config.base_accuracy}")
            base_accuracy = self.model_trainer_config.base_accuracy
            
            logging.info('Initializing operation model selection')
            best_model = model_factory.get_best_model(X=X_train, y=y_train, base_accuracy=base_accuracy)
            
            logging.info(f"Best model found for train dataset : {best_model}")
            
            logging.info('Extracting the trained model list')
            grid_search_models_list : List[GridSearchBestModel] = model_factory.grid_search_best_models_list
            
            model_list = [model.best_model for model in grid_search_models_list]
            logging.info('Evaluating all trained models on training and testing datasets both')
            
            metric_info : MetricInfoArtifact = evaluate_regression_model(
                model_list=model_list,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                base_accuracy=base_accuracy
            )
            
            logging.info(f'Best model found for both training and testing dataset.')
            preprocessed_obj = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            
            model_obj = metric_info.model_object
            
            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            
            logging.info(f"Saving the model at file path : {trained_model_file_path}")
            
            black_friday_model = BlackFridayEstimatorModel(
                preprocessing_obj=preprocessed_obj,
                trained_model_obj=model_obj
            )
            save_object(file_path=trained_model_file_path, object=black_friday_model)
            
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=trained_model_file_path,
                is_trained=True,
                message="Model trainer Performed Sucessfully",
                train_accuracy_score=metric_info.train_accuracy,
                test_accuracy_score=metric_info.test_accuracy,
                train_rmse=metric_info.train_rmse,
                test_rmse=metric_info.test_rmse,
                model_accuracy=metric_info.model_accuracy
            )
            
            logging.info(f'Model Trainer Artifact: {model_trainer_artifact}')
            return model_trainer_artifact
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def __del__(self) -> str:
        logging.info(f"{'>>' * 20} Model Trainer log is completed {'<<' * 20}")