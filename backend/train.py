import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bar

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
dataset = datasets.ImageFolder(root="dataset_folder", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load a pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification (real/fake)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print the device being used
print(f"Using device: {device}")

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    # Wrap the dataloader with tqdm for a progress bar
    with tqdm(dataloader, unit="batch") as tepoch:
        for inputs, labels in tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

            # Move data to the GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Update the progress bar with the current loss
            tepoch.set_postfix(loss=loss.item())

    # Print average loss for the epoch
    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {epoch_loss}")

# Save the trained model
torch.save(model.state_dict(), "deepfake_detection_model.pth")