import os
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
import torch

class FilteredImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except PIL.UnidentifiedImageError:
            print(f"Skipping invalid image at index {index}")
            return None

# Use the custom dataset class

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False

# Override default loader
def custom_loader(path):
    if is_valid_image(path):
        return Image.open(path)
    else:
        raise PIL.UnidentifiedImageError(f"Invalid image file: {path}")
datasets.ImageFolder.loader = custom_loader
def scan_dataset(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not is_valid_image(file_path):
                print(f"Invalid image found: {file_path}")

# Scan dataset directory
scan_dataset('dataset/train')
scan_dataset('dataset/val')
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Current CUDA device:", torch.cuda.current_device())
print("CUDA version:", torch.version.cuda)
print("PyTorch version:", torch.__version__)