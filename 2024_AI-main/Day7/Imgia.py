import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root="./data/", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./data/", train=False, download=True, transform=transform)

# Create data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)

# Define classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def convert_to_imshow_format(image):
    # CHW => HWC
    image = image / 2 + 0.5  # Corrected normalization
    image = image.numpy()
    image = image.transpose((1, 2, 0))
    image = image * 255
    image = image.astype(np.uint8)
    return image

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # CIFAR10

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define the training function
def train():
    # Load CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # Initialize the neural network
    net = Net()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    epochs = 2
    model_dir_path = "model/"
    model_path = f"{model_dir_path}cifar-10-cnn-model.pth"

    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path))
    else:
        # Do training
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print("Training Done")
        torch.save(net.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    return net

if __name__ == '__main__':
    net = train()

# Step 4

dataiter = iter(testloader)
images, labels = next(dataiter)

fig, ax = plt.subplots(1, len(images), figsize=(12, 2.5))

for idx, image in enumerate(images):
    ax[idx].imshow(convert_to_imshow_format(image))
    ax[idx].set_title(classes[labels[idx].item()])
    ax[idx].set_xticks([])
    ax[idx].set_yticks([])

outputs = net(images)
_, predicted = torch.max(outputs, 1)

softmax = nn.Softmax(dim=1)
probs = softmax(outputs)

for idx, output in enumerate(outputs):
    print(f"Predicted: {classes[predicted[idx].item()]}    |    Probability: {probs[idx][predicted[idx]].item()}")

plt.show()
