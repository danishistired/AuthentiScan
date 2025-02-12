import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from mtcnn import MTCNN

# Load the trained model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification (real/fake)
model.load_state_dict(torch.load("deepfake_detection_model.pth"))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize MTCNN for face detection
detector = MTCNN()

def predict(image_path):
    # Detect face in the image
    image = cv2.imread(image_path)
    faces = detector.detect_faces(image)
    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        face_img = image[y:y+height, x:x+width]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = Image.fromarray(face_img)

        # Preprocess and predict
        input_tensor = transform(face_img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            print("Fake" if predicted.item() == 0 else "Real")

# Example usage
predict("static/test_image2.jpg")