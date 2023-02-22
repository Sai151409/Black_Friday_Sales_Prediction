from black_friday.exception import BlackFridayException
import os, sys
import yaml
import pandas as pd
import numpy as np
import dill



def read_yaml_file(file_path : str):
    try:
        with open(file=file_path, mode="rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise BlackFridayException(e, sys) from e
    
def write_yaml_file(file_path : str, data : dict = None):
    """Create a yaml file

    Args:
        file_path (str): str
        data (dict, optional): dict
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, mode="w") as yaml_file:
            if data is not None:
                yaml.dump(data, yaml_file)
    except Exception as e:
        raise BlackFridayException(e, sys) from e
    
def load_data(file_path:str, schema_file_path:str):
    try:
        data_frame = pd.read_csv(file_path)
        schema = read_yaml_file(schema_file_path)
        columns = data_frame.columns
        for i in columns:
            if i in schema['columns'].keys():
                data_frame[i].astype(schema['columns'][i])
            else:
                raise Exception("Column is not in the schema")
            
        return data_frame
    except Exception as e:
        raise BlackFridayException(e, sys) from e
    
def save_numpy_array(file_path: str, array : np.array):
    """save numpy array

    Args:
        file_path (str): file path where we have to save the numpy object
        object (_type_): numpy object
    """
    try:
        dirname = os.path.dirname(file_path)
        os.makedirs(dirname, exist_ok=True)
        with open(file_path, "wb") as file:
            np.save(file, array)
    except Exception as e:
        raise BlackFridayException(e, sys) from e
    
def load_numpy_array(file_path: str) -> np.array:
    """load numpy array

    Args:
        file_path (str): file path where we have to save the numpy object
    """
    try:
        with open(file_path, "rb") as file:
            return np.load(file)
    except Exception as e:
        raise BlackFridayException(e, sys) from e
    
def save_object(file_path: str, object):
    """save object

    Args:
        file_path (str): file path where we have to save the object
        object (_type_): object
    """
    try:
        dirname = os.path.dirname(file_path)
        os.makedirs(dirname, exist_ok=True)
        with open(file_path, "wb") as file:
            dill.dump(object, file)
    except Exception as e:
        raise BlackFridayException(e, sys) from e
    
def load_object(file_path: str) :
    """load object

    Args:
        file_path (str): object file path 
    """
    try:
        with open(file_path, "rb") as file:
            return dill.load(file=file)
    except Exception as e:
        raise BlackFridayException(e, sys) from e