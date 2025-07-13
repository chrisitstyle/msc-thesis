import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Initialize the Kaggle API
api = KaggleApi()
api.authenticate()

# Download the dataset
dataset = 'ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes'
download_path = './data'
api.dataset_download_files(dataset, path=download_path, unzip=False)

# Unzip the downloaded ZIP file
zip_path = os.path.join(download_path, 'mri-for-brain-tumor-with-bounding-boxes.zip')
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(download_path)

# Delete the ZIP file after extraction
os.remove(zip_path)

print("Dataset has been downloaded and extracted.")