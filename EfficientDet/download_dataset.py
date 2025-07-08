import os
import kaggle

DATASET = 'ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes'
DEST = 'data'

def download_and_extract():
    os.makedirs(DEST, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET, path=DEST, unzip=True)
    print("✅ Dataset downloaded and extracted to:", DEST)
    print_summary()

def print_summary():
    """
    After extraction, walk through Train/Val splits and for each class
    print the number of images and label files.
    """
    for split in ['Train', 'Val']:
        split_dir = os.path.join(DEST, split)
        if not os.path.isdir(split_dir):
            print(f"⚠️  Directory not found: {split_dir}")
            continue

        print(f"\n{split} Set:")
        for cls_name in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls_name)
            img_dir = os.path.join(cls_dir, 'images')
            lbl_dir = os.path.join(cls_dir, 'labels')

            num_images = count_files(img_dir)
            num_labels = count_files(lbl_dir)

            print(f"- {cls_name}: {num_images} images, {num_labels} labels")

def count_files(directory):
    """
    Return the number of files in the given directory.
    If the directory does not exist, returns 0.
    """
    if not os.path.isdir(directory):
        return 0
    return sum(
        1 for entry in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, entry))
    )

if __name__ == '__main__':
    download_and_extract()
