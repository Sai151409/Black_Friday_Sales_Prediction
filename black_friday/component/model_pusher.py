from black_friday.logger import logging 
from black_friday.exception import BlackFridayException
import os, sys
from black_friday.entity.config_entity import ModelPusherConfig
from black_friday.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact
import shutil

class ModelPusher:
    def __init__(self, model_evaluation_artifact : ModelEvaluationArtifact,
                 model_pusher_config : ModelPusherConfig):
        try:
            logging.info(f'{">>" * 20} Model Pusher log is Started {"<<" * 20}')
            self.model_evaluation_artifact = model_evaluation_artifact
            self.model_pusher_config = model_pusher_config
        except Exception as e:
            raise BlackFridayException(e, sys) from e
    
    def export_module(self) -> ModelPusherArtifact:
        try:
            evaluation_model_file_path = self.model_evaluation_artifact.evaluated_model_path
            export_dir = self.model_pusher_config.export_dir_path
            filename = os.path.basename(evaluation_model_file_path)
            export_file_path = os.path.join(export_dir, filename)
            logging.info(f"Exporting the evaluated model from {evaluation_model_file_path} to {export_dir}")
            os.makedirs(export_dir, exist_ok=True)
            shutil.copy(src = evaluation_model_file_path, dst=export_file_path)
            logging.info(f"Sucessfully copied the file {evaluation_model_file_path} to {export_file_path}")
            model_pusher_artifact = ModelPusherArtifact(
                is_model_pushed=True,
                export_model_file_path=export_file_path
            )
            logging.info(f"Model Pusher Artifact : {model_pusher_artifact}")
            return model_pusher_artifact
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            self.export_module()
        except Exception as e:
            raise BlackFridayException(e, sys) from e
        
    def __del__(self):
        logging.info(f"{'>>' * 20} Model Pusher log is completed {'<<' * 20}")