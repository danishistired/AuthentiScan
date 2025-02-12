import torch

MODEL_PATH = "deepfake_detection_model.pth"

# Load the model file
model_data = torch.load(MODEL_PATH, map_location="cpu")

# Check if it's a full model or just state_dict
if isinstance(model_data, dict):
    print("Keys in model file:", model_data.keys())  # Check available keys
else:
    print("Loaded model:", model_data)  # Print the model structure
