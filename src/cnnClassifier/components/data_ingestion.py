import os
import zipfile
from pathlib import Path

import gdown
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self) -> None:
        """
        Download file from Google Drive
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = Path(self.config.local_data_file)
            os.makedirs(self.config.root_dir, exist_ok=True)
            logger.info(f"Downloading file from :[{dataset_url}] into :[{zip_download_dir}]")
            
            file_id = dataset_url.split('/')[-2]
            prefix = "https://drive.google.com/uc?export=download&id="
            gdown.download(prefix + file_id, str(zip_download_dir))
            
            logger.info(f"Downloaded file from :[{dataset_url}] into :[{zip_download_dir}] of size :[{get_size(zip_download_dir)}]")
            
        except Exception as e:
            raise e
        
    def extract_zip_file(self) -> None:
        """
        Extract zip file
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)