from torchvision import models, transforms
import torch

# Load a pre-trained ResNet model for feature extraction
model = models.resnet50(pretrained=True)
model.eval()

# Transformation pipeline for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image):
    # Transform the image and add a batch dimension
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image).flatten()
    return features
