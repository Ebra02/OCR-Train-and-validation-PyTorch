import os
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torchvision import models  # Import VGG

# Set a specific random seed
random_seed = 42

# Set the random seed for Python's random module
random.seed(random_seed)

# Set the random seed for NumPy
np.random.seed(random_seed)

# Set the random seed for PyTorch
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Specify the directory containing your images
data_dir = r"C:\Users\Ebra\OneDrive\سطح المكتب\Projects\OCR-Train-and-validation-PyTorch//SmallDataset"

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize the input images to 64x64
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load your custom dataset from the directory
custom_dataset = ImageFolder(root=data_dir, transform=transform)

# Split the dataset into training and testing sets
train_size = int(0.8 * len(custom_dataset))
test_size = len(custom_dataset) - train_size
train_dataset, test_dataset = random_split(custom_dataset, [train_size, test_size])

# Create data loaders for training and testing
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define VGGNet and move it to the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=False, num_classes=len(custom_dataset.classes)).to(device)

# Define the optimizer and criterion
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training hyperparameters
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

accuracy = 100 * correct / total
print(f"Testing Accuracy for VGGNet: {accuracy:.2f}%")
