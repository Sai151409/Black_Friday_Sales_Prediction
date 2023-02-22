from black_friday.logger import logging
from black_friday.exception import BlackFridayException
import os, sys
import importlib
import yaml
from collections import namedtuple
from typing import List
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


GRID_SEARCH_KEY = "grid_search"
CLASS_KEY = 'class'
MODULE_KEY = 'module'
MODULE_SELECTION_KEY = 'model_selection'
PARAMS_KEY = 'params' 
SEARCH_PARAM_GRID_KEY = "search_param_grid"

InitializedModelDetail = namedtuple('IntializedModelDetai', 
                                   ['model_serial_number',
                                    'model',
                                    'param_grid_search', 
                                    'model_name'])

GridSearchBestModel = namedtuple(
    'GridSearchModel', [
        'model_serial_number',
        'model',
        'best_model',
        'best_parameter', 
        'best_score'
    ]
)

BestModel = namedtuple('BestModel', [
    'model_serial_number',
    'model',
    'best_model',
    'best_parameter',
    'best_score'
])

MetricInfoArtifact = namedtuple('MetricInfoArtifact',
    [
        'model_name',
        'model_object',
        'train_rmse',
        'test_rmse',
        'train_accuracy', 
        'test_accuracy',
        'model_accuracy',
        'index_number'
    ]
)

def evaluate_regression_model(model_list : list, 
                              X_train : np.ndarray,
                              y_train : np.ndarray,
                              X_test : np.ndarray,
                              y_test : np.ndarray,
                              base_accuracy : float = 0.6) -> MetricInfoArtifact:
    """
    Description:
    This function compare multiple linear regerssion model return the best model

    Params:
        model_list (list): list of models
        X_train (np.ndarray): Training dataset input feature
        y_train (np.ndarray): Training dataset target feature
        X_test (np.ndarray): Testing dataset input feature
        y_test (np.ndarray): Testing dataset target feature
        base_accuracy (float, optional): _description_. Defaults to 0.6.

    Returns:
    It returns a namedtuple
        MetricInfoArtifact = namedtuple("MetricInfoArtifact",
                                ["model_name", "model_object", "train_rmse", "test_rmse", "train_accuracy",
                                 "test_accuracy", "model_accuracy", "index_number"]) 
    """
    try:
        index= 0 
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model) # getting model name based on model object
            logging.info(f"{'>>' * 30} Started evaluating model : [{type(model).__name__}] {'<<' * 30}")
            
            #Getting prediction for training and testing  dataset
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_acc = r2_score(y_train, y_train_pred)
            test_acc = r2_score(y_test, y_test_pred)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            model_accuracy = (2 * train_acc * test_acc) / (train_acc + test_acc)
            diff_test_train_acc = abs(train_acc - test_acc)
            
            # logging all important metric
            logging.info(f"{'>>' * 30} Score {'<<' * 30}") 
            logging.info(f"Train Score \t\t Test Score \t\t Average Score")
            logging.info(f"{train_acc} \t\t {test_acc} \t\t {model_accuracy}")
            
            logging.info(f"{'>>' * 30} Loss {'<<' * 30}")
            logging.info(f"Diff test and train : {diff_test_train_acc}")
            logging.info(f"Train root mean squared error : {train_rmse}")
            logging.info(f"Testing root mean squared error : {test_rmse}")
            #if model accuracy is greater than base accuracy and train and test score is within certain thershold
            #we will accept that model as accepted model
            
            if model_accuracy >= base_accuracy and diff_test_train_acc < 0.05:
                base_accuracy = model_accuracy
                metric_info_artifact = MetricInfoArtifact(
                    model_name=model_name,
                    model_object=model,
                    train_accuracy=train_acc,
                    test_accuracy=test_acc,
                    train_rmse=train_rmse,
                    test_rmse=test_rmse,
                    model_accuracy=model_accuracy,
                    index_number=index    
                )
                logging.info(f"Acceptable model found : {metric_info_artifact}")
            index += 1
            
        if metric_info_artifact is None:
            logging.info(f"No model is higher accuracy than the base accuracy")
        return metric_info_artifact
    except Exception as e:
        raise BlackFridayException(e, sys) from e 
                              
def get_sample_yaml_file(export_dir):
    try: 
        model_config = {
            "GRID_SEARCH_KEY" :
                {
                    CLASS_KEY : "GridSearchCV",
                    MODULE_KEY : "sklearn.model_selection",
                    PARAMS_KEY : 
                        {
                        "cv" : 3,
                        "verbose" : 1
                        }
                },
            "MODEL_SELECTION" : 
                {
                    "module 0" :
                        {
                            MODULE_KEY : "module_of_model",
                            CLASS_KEY : "ModelClassName",
                            PARAMS_KEY : 
                                {
                                    "param_name1" : "value_1",
                                    "param_name2" : "value_2"
                                },
                            SEARCH_PARAM_GRID_KEY :
                                {
                                    "param_name": ['param_value_1', 'param_value_2']
                                }
                        },
                        
                }
                
        }
        
        os.makedirs(export_dir, exist_ok=True)
        export_file_path = os.path.join(export_dir, "model.yaml")
        with open(export_file_path, "w") as file:
            yaml.dump(model_config, file)
        return export_file_path
    except Exception as e:
        raise BlackFridayException(e, sys) from e

class ModelFactory:
    def __init__(self, model_config_path = r'C:\Users\Asus\ML_Projects\Black_Friday_Sales_Prediction\black_friday\entity\model_factory.py'):
        try:
            self.config : dict = ModelFactory.read_params(model_config_path)
            self.model_config : str = self.config[GRID_SEARCH_KEY]
            self.grid_search_module : str = self.model_config[MODULE_KEY]
            self.grid_search_class : str = self.model_config[CLASS_KEY]
            self.grid_search_property_data : dict = self.model_config[PARAMS_KEY]
            self.models_initialization_config : dict = self.config[MODULE_SELECTION_KEY]
            
            self.initialized_models_list = None
            self.grid_search_best_models_list = None
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    @staticmethod
    def update_property(instance_ref : object, property_data:dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception("Property data parameter required to dictionary")
            for key, value in property_data.items():
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    @staticmethod
    def class_for_name(module_name, class_name):
        try:
            module = importlib.import_module(module_name)
            logging.info(f"Executing the commmand : import {module_name} from {class_name}")
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    @staticmethod
    def read_params(file_path:str):
        try:
            with open(file_path, "rb") as file:
                config :dict = yaml.safe_load(file)
                return config
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def execute_grid_search_operations(self, initialized_model : InitializedModelDetail,
                                       input_feature, output_feature) -> GridSearchBestModel:
        """
        excute_grid_search_operation(): function will perform paramter search operation and
        it will return you the best optimistic  model with best paramter:
        estimator: Model object
        param_grid: dictionary of paramter to perform search operation
        input_feature: your all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return GridSearchOperation object
        """
        try:
            grid_search_cv_ref = ModelFactory.class_for_name(
                module_name=self.grid_search_module,
                class_name=self.grid_search_class
            )
            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                param_grid=initialized_model.param_grid_search)
            grid_search_cv = ModelFactory.update_property(
                instance_ref=grid_search_cv,
                property_data=self.grid_search_property_data
            )
            
            message = f"{'==' * 30} Training {type(initialized_model.model).__name__} started {'==' * 30}"
            logging.info(message)
            grid_search_cv.fit(input_feature, output_feature)
            message = f"{'==' * 30} Training {type(initialized_model.model).__name__} completed {'==' * 30}"
            logging.info(message)
            
            grid_search_best_model = GridSearchBestModel(
                model=initialized_model.model,
                model_serial_number=initialized_model.model_serial_number,
                best_model=grid_search_cv.best_estimator_,
                best_parameter=grid_search_cv.best_params_,
                best_score=grid_search_cv.best_score_
            )
            
            return grid_search_best_model
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def get_initialized_models_list(self) -> List[InitializedModelDetail]:
        try:
            initialized_models_list = []
            print(self.models_initialization_config.keys())
            for model_serial_number in self.models_initialization_config.keys():
                model_initialization_config = self.models_initialization_config[model_serial_number]
                model_obj_ref = ModelFactory.class_for_name(
                    module_name=model_initialization_config[MODULE_KEY],
                    class_name=model_initialization_config[CLASS_KEY]
                )
                model = model_obj_ref()
                if PARAMS_KEY in model_initialization_config:
                    model = ModelFactory.update_property(
                        instance_ref=model,
                        property_data=model_initialization_config[PARAMS_KEY]
                    )
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"
                
                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                
                initialized_model = InitializedModelDetail(
                    model_serial_number=model_serial_number,
                    model = model,
                    model_name=model_name,
                    param_grid_search=param_grid_search
                )
                
                initialized_models_list.append(initialized_model)
                
            self.initialized_models_list = initialized_models_list
            
            return self.initialized_models_list
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def get_grid_search_models_list(self, initialized_model_list : List[InitializedModelDetail],
                                    input_feature,
                                    output_feature):
        try:
            grid_search_best_models_list = []
            for initialized_model in initialized_model_list:
                grid_search_best_model = self.execute_grid_search_operations(
                    initialized_model=initialized_model,
                    input_feature=input_feature,
                    output_feature=output_feature
                )
                grid_search_best_models_list.append(grid_search_best_model)
            self.grid_search_best_models_list = grid_search_best_models_list
            return self.grid_search_best_models_list
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    @staticmethod
    def get_model_details(self, model : List[InitializedModelDetail],
                          model_serial_number):
        try:
            for model_data in model:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
        except Exception as e:
            raise BlackFridayException(e, sys) from e
    
    @staticmethod    
    def get_best_model_from_grid_search_best_models_list(
        grid_search_best_models_list : List[GridSearchBestModel],
        base_accuracy = 0.6) -> BestModel:
        try:
            best_model = None
            for grid_search_best_model in grid_search_best_models_list:
                if base_accuracy < grid_search_best_model.best_score:
                    logging.info(f"Acceptable model found : {grid_search_best_model}")
                    base_accuracy = grid_search_best_model.best_score
                    
                    best_model = grid_search_best_model
            if not best_model:
                raise Exception(f"None of the model has base accuracy : {base_accuracy}")
            logging.info(f"Best Model : {best_model}")  
            return best_model
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def get_best_model(self, X, y, base_accuracy) -> BestModel:
        try:
            logging.info('Started Initialized model from config file')
            intialized_models_list = self.get_initialized_models_list()
            grid_search_best_models_list = self.get_grid_search_models_list(
                initialized_model_list=intialized_models_list,
                input_feature=X,
                output_feature=y    
            )
            
            return ModelFactory.get_best_model_from_grid_search_best_models_list(
                grid_search_best_models_list=grid_search_best_models_list,
                base_accuracy=base_accuracy
            )
        except Exception as e:
            raise BlackFridayException(e, sys) from e
            