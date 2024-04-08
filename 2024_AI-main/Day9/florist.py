import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import torchvision.models as models
import torch.optim as optim

# Checking if GPU is available and setting device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU is not available")

# Read DataSet
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = torchvision.datasets.ImageFolder("flowers", transform=transform)
# Use a larger batch size for more stable training
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

class CNNFlower(nn.Module):
    def __init__(self):
        super(CNNFlower, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)  # half
        # Do tests
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # linear layers
        self.lin_size = 512 * 8 * 8  # Update this calculation based on the output size after convolutions
        self.fc1 = nn.Linear(self.lin_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1024)
        self.fc4 = nn.Linear(1024, 100)
        self.out = nn.Linear(100, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = x.view(-1, self.lin_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.out(x)
        return x

# Initialize the model and move it to the device
Net = CNNFlower().to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(Net.parameters(), lr=0.001, momentum=0.9)

# Directory for saving model
model_dir_path = "model/"
if not os.path.exists(model_dir_path):
    os.makedirs(model_dir_path)

modelname = "flower_model.pth"
if os.path.exists(f"{model_dir_path}{modelname}"):
    Net.load_state_dict(torch.load(f"{model_dir_path}{modelname}"))

best_loss = float('inf')

# Put the model in training mode
Net.train()

# Training loop
for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = Net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f"Epoch {epoch + 1}, Iteration {i + 1}, Loss: {running_loss / 100}")
            if running_loss < best_loss:
                best_loss = running_loss
                torch.save(Net.state_dict(), f"{model_dir_path}{modelname}")
            running_loss = 0.0  # Reset running loss
