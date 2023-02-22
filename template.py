import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]-[%(message)s]')

package_name = 'black_friday'

file_list = [
    'github/workflows/main.yaml',
    'config/config.yaml',
    'config/model.yaml',
    'config/schema.yaml',
    f'{package_name}/__init__.py',
    f'{package_name}/component/__init__.py',
    f'{package_name}/component/data_ingestion.py',
    f'{package_name}/component/data_validation.py',
    f'{package_name}/component/data_transformation.py',
    f'{package_name}/component/model_trainer.py',
    f'{package_name}/component/model_evaluation.py',
    f'{package_name}/component/model_pusher.py',
    f'{package_name}/config/__init__.py',
    f'{package_name}/config/configuartion.py',
    f'{package_name}/constant/__init__.py',
    f'{package_name}/entity/__init__.py',
    f'{package_name}/entity/artifact_entity.py',
    f'{package_name}/entity/config_entity.py',
    f'{package_name}/entity/black_friday_predictor.py',
    f'{package_name}/entity/model_factory.py',
    f'{package_name}/exception/__init__.py',
    f'{package_name}/logger/__init__.py',
    f'{package_name}/pipeline/__init__.py',
    f'{package_name}/pipeline/pipeline.py',
    f'{package_name}/util/__init__.py',
    f'{package_name}/util/util.py',
    '.dockerignore',
    'docker-compose.yaml',
    'Dockerfile',
    'requirements.txt',
    'setup.py',
    'Analysis/analysis.ipynb'
]

for file_path in file_list:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f'Creating directory {file_dir} for file : {file_name}')
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, 'w'):
            pass
            logging.info(f'Creating empty file : {file_name}')
    else:
        logging.info(f'{file_name} has already exists')