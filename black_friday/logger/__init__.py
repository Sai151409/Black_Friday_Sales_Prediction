import logging
import os, sys
from datetime import datetime
from black_friday.exception import BlackFridayException
import pandas as pd

ROOT_DIR = os.getcwd()

LOG_DIR = "black_friday_logs"

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE_NAME = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'

LOG_FILE_PATH = os.path.join(ROOT_DIR, LOG_DIR, LOG_FILE_NAME)

logging.basicConfig(filename=LOG_FILE_PATH,
                    filemode='w', level=logging.INFO,
                    format="[%(asctime)s] - %(levelname)s - %(lineno)s - %(filename)s - %(funcName)s() - %(message)s")

def get_log_dataframe(file_path : str):
    """
    It returns the dataframe of credit card logs

    Args:
        filepath (str): log file path
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            data.append(line)
            
    log_data = pd.DataFrame(data, columns = ['Timestamp', 'log_level', 'error_lineno',
                                             'error_filename', 'error_funcname', 'message'])
    
    log_data['error_message'] = log_data['Timestamp'].astype(str) + ':$' + log_data['message']
    
    return log_data['error_message']