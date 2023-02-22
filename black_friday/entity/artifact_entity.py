from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact",[
    "train_file_path", 'test_file_path', 'is_ingested', 'message'
])

DataValidationArtifact = namedtuple('DataValidationArtifact', [
    "validated_train_file_path", "validated_test_file_path", "is_validated", 'message',
    'schema_file_path', 'report_file_path', 'report_page_file_path'
])

DataTransformationArtifact = namedtuple("DataTransformationArtifact", [
    'is_transformed', 'message', 'transformed_train_file_path',
    'transformed_test_file_path', 'preprocessed_object_file_path'
])

ModelTrainerArtifact = namedtuple("ModelTrainerArtifact", [
    "trained_model_file_path", 'is_trained', 'message',
    "train_accuracy_score", "test_accuracy_score", "model_accuracy",
    "train_rmse", "test_rmse"
])

ModelEvaluationArtifact = namedtuple("ModelEvaluationArtifact", [
    "is_model_accepted", "evaluated_model_path"
])

ModelPusherArtifact = namedtuple("ModelPusherArtifact", [
    "is_model_pushed", "export_model_file_path"
])