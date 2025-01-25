from pathlib import Path
from typing import List
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from torchvision import models
import numpy as np


class DeepFakeDetector(nn.Module):
    def __init__(self):
        super(DeepFakeDetector, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.8),  # Dropout used in training
            nn.Linear(self.model.fc.in_features, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(1)


def get_test_image_paths():
    # normal pictures are here
    nature_dir = Path("dataset/val/nature")
    midjourney_dir = Path("dataset/val/midjourney")

    nature_images = sorted(nature_dir.glob("*.png"))[:50]  # Adjust extension if needed
    midjourney_images = sorted(midjourney_dir.glob("*.png"))[:50]  # Adjust extension if needed

    test_images = nature_images + midjourney_images

    print("Selected test image paths:", test_images)
    return test_images


def classify_images(img_paths: List[Path]) -> List[bool]:
    """
    Classify whether each image in the provided list is computer-generated or not.

    Args:
        img_paths (List[Path]): A list of paths to the images to be classified.
    Returns:
        List[bool]: A list of boolean values indicating whether each image is computer-generated.
                    True indicates the image is computer-generated, while False indicates it is not.
    """
    model = DeepFakeDetector()
    try:
        # Load the model weights
        state_dict = torch.load("deepfake_detector_best_0958_needs_to_be_tested.pth", map_location=torch.device('cpu'))
        new_state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
        model.model.load_state_dict(new_state_dict)

    except Exception as e:
        print(f"Error loading state_dict: {e}")
        return []

    model.eval()  # Set model to evaluation mode

    # Transform pipeline for the magnitude spectrum (adjust as per training normalization)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to match training input
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ResNet normalization
    ])

    predictions = []

    for img_path in img_paths:
        try:
            # Load the image
            image = Image.open(img_path).convert("L")  # Convert to grayscale
            
            # Convert the image to a numpy array
            image_array = np.array(image)
            
            # Perform FFT on the image
            fft_result = np.fft.fft2(image_array)
            fft_shifted = np.fft.fftshift(fft_result)
            magnitude_spectrum = 20 * np.log(np.abs(fft_shifted))
            spectrum_image = Image.fromarray(magnitude_spectrum).convert("RGB")
            input_tensor = transform(spectrum_image).unsqueeze(0)

            with torch.no_grad():
                logits = model(input_tensor)
                probabilities = torch.sigmoid(logits)
                print(img_path)
                print(probabilities.item())
                prediction = probabilities.item() < 0.999796
                predictions.append(prediction)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            predictions.append(False)

    return predictions
def main():
    # Get the list of test image paths
    test_image_paths = get_test_image_paths()

    predictions = classify_images(test_image_paths)
    print("Predictions:", predictions)
    
if __name__ == "__main__":
    main()