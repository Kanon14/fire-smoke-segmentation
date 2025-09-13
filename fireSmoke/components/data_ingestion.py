import os
import sys
import zipfile
import gdown
from fireSmoke.logger import logging
from fireSmoke.exception import AppException
from fireSmoke.entity.config_entity import DataIngestionConfig
from fireSmoke.entity.artifacts_entity import DataIngestionArtifact


class DataIngestion:
    """
    This class handles the data ingestion process, including downloading the dataset 
    from a specified URL, extracting the downloaded zip file, and preparing it for 
    further processing.
    """
    
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        Constructor for the DataIngestion class.
        
        :param data_ingestion_config: Configuration for data ingestion including 
                                      download URL and directory paths.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
           raise AppException(e, sys)
        
        
    def download_data(self) -> str:
        """
        Downloads the dataset from a specified URL to a local directory.

        :return: Path to the downloaded zip file.
        :raises AppException: If an error occurs during the download process.
        """
        try:
            dataset_url = self.data_ingestion_config.data_download_url
            zip_download_dir = self.data_ingestion_config.data_ingestion_dir
            os.makedirs(zip_download_dir, exist_ok=True)
            data_file_name = "data.zip"
            zip_file_path = os.path.join(zip_download_dir, data_file_name)
            logging.info(f"Downloading data from {dataset_url} into the {zip_file_path}")
            
            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="
            gdown.download(prefix+file_id, zip_file_path)
            
            logging.info(f"Downloaded data from {dataset_url} into file {zip_file_path}")
            
            return zip_file_path
        
        except Exception as e:
            raise AppException(e, sys)
        
        
    def extract_zip_file(self, zip_file_path: str) -> str:
        """
        Extracts the downloaded zip file into a specified directory.
        
        :param zip_file_path: Path to the zip file to be extracted.
        :return: Path to the directory where files are extracted.
        :raises AppException: If an error occurs during the extraction process.
        """
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(feature_store_path, exist_ok=True)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(feature_store_path)
            logging.info(f"Extracting zip file: {zip_file_path} into dir: {feature_store_path}")
            
            return feature_store_path
        
        except Exception as e:
            raise AppException(e, sys)
        
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Orchestrates the data ingestion process by downloading and extracting the dataset.
        
        :return: DataIngestionArtifact containing paths to the zip file and extracted data directory.
        :raises AppException: If an error occurs during the data ingestion process.
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")
        try:
            zip_file_path = self.download_data()
            feature_store_path = self.extract_zip_file(zip_file_path)
            
            data_ingestion_artifact = DataIngestionArtifact(
                data_zip_file_path = zip_file_path,
                feature_store_path = feature_store_path
            )
            
            logging.info("Exited initiate_data_ingestion method of Data_Ingestion class")
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            
            return data_ingestion_artifact
        
        except Exception as e:
            raise AppException(e, sys)