from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict":{"origins":"*"}})  # Allow frontend to communicate with backend





# ✅ Define the same model architecture as in training
def load_model(model_path):
    model = models.resnet50(weights=None)


    model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification (Real/Fake)

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))  # Load weights
    model.eval()  # Set to evaluation mode
    return model


# ✅ Load the AI model
MODEL_PATH = "deepfake_detection_model.pth"  # Ensure this file exists
model = load_model(MODEL_PATH)

# ✅ Define preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # ✅ Perform inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()

    result = "Real" if prediction == 1 else "Deepfake"
    return jsonify({'result': result})

@app.route('/')
def home():
    return "Flask Server is Running. Use /predict to test image uploads."



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

