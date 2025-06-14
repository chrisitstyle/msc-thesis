import os
import zipfile
import kaggle

DATASET = 'ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes'
DEST = 'data'

def download_and_extract():
    os.makedirs(DEST, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET, path=DEST, unzip=True)
    print("✅ Dataset downloaded and extracted to:", DEST)

if __name__ == '__main__':
    download_and_extract()
