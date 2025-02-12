import torch
print(torch.cuda.is_available())  # Should return True if GPU is detected
print(torch.cuda.get_device_name(0))  # Should return the name of your GPU (e.g., "NVIDIA GeForce RTX 4060")