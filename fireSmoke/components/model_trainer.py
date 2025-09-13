import os, sys
from fireSmoke.logger import logging
from fireSmoke.exception import AppException
from fireSmoke.entity.config_entity import ModelTrainerConfig
from fireSmoke.entity.artifacts_entity import ModelTrainerArtifact
from ultralytics import YOLO


class ModelTrainer:
    """
    This class handles the training of the YOLO model for object detection.
    It manages data preparation, model training, and cleanup processes.
    """
    
    def __init__(self, 
                 model_trainer_config: ModelTrainerConfig):
        """
        Constructor for the ModelTrainer class.
        
        :param model_trainer_config: Configuration object containing training parameters like
                                     weights, batch size, image size, and number of epochs.
        """
        self.model_trainer_config = model_trainer_config
        
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Orchestrates the entire model training process, including:
        - Unzipping and preparing data
        - Training the YOLO model
        - Saving the best-trained model
        - Cleaning up temporary files

        :return: ModelTrainerArtifact containing the path to the best-trained model.
        :raises AppException: If any step of the process fails.
        """
        logging.info("Entered intiate_model_trainer method of ModelTrainer class")
        
        try:
            # Unzipping and preparing data
            logging.info("Unzipping data")
            os.system("unzip data.zip -d yolo_seg_train")
            os.system("rm data.zip")
            
            # Ensure the training directory exists
            os.makedirs("yolo_seg_train", exist_ok=True)
            
            # Path to the data.yaml which should be relative or configured externally
            data_yaml_path = os.path.abspath("yolo_seg_train/data.yaml")
            
            # Running the training process
            os.system(f"cd yolo_seg_train && yolo task=segment mode=train \
                    model={self.model_trainer_config.weight_name} \
                    imgsz={self.model_trainer_config.img_size} \
                    batch={self.model_trainer_config.batch_size} \
                    epochs={self.model_trainer_config.no_epochs} \
                    data={data_yaml_path} \
                    name='{self.model_trainer_config.weight_name.split('.')[0]}'")
            
            # Path for saving the best model
            best_model_path = f"yolo_seg_train/runs/detect/{self.model_trainer_config.weight_name.split('.')[0]}/weights/best.pt"
            os.system(f"cp {best_model_path} yolo_seg_train/")
            
            # Ensure the model trainer directory exists and copy the best model
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
            os.system(f"cp {best_model_path} {self.model_trainer_config.model_trainer_dir}/")
            
            # Cleanup to remove unnecessary files and directories
            os.system(f"rm -rf yolo_seg_train/{self.model_trainer_config.weight_name}")
            os.system("rm -rf yolo_seg_train/runs") 
            os.system("rm -rf yolo_seg_train/train")
            os.system("rm -rf yolo_seg_train/test")
            os.system("rm -rf yolo_seg_train/valid")
            os.system("rm -rf yolo_seg_train/data.yaml")
            os.system("rm -rf yolo_seg_train/README.dataset.txt")
            os.system("rm -rf yolo_seg_train/README.roboflow.txt")
            
            # Creating artifact object for the trained model
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=os.path.join(self.model_trainer_config.model_trainer_dir, "best.pt")
            )
            
            logging.info("Exited initiate_model_trainer method of ModelTrainer class")
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact
            
        except Exception as e:
            raise AppException(e, sys)