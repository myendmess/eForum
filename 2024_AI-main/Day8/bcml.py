###############################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
###############################################################
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
###############################################################
# Loading the dataset
wdbc_path = "wdbc.data"  
column_names = [
    "ID", "Diagnosis", "Mean Radius", "Mean Texture", "Mean Perimeter", "Mean Area", 
    "Mean Smoothness", "Mean Compactness", "Mean Concavity", "Mean Concave Points", 
    "Mean Symmetry", "Mean Fractal Dimension", "SE Radius", "SE Texture", "SE Perimeter", 
    "SE Area", "SE Smoothness", "SE Compactness", "SE Concavity", "SE Concave Points", 
    "SE Symmetry", "SE Fractal Dimension", "Worst Radius", "Worst Texture", "Worst Perimeter", 
    "Worst Area", "Worst Smoothness", "Worst Compactness", "Worst Concavity", "Worst Concave Points", 
    "Worst Symmetry", "Worst Fractal Dimension"
]
df = pd.read_csv(wdbc_path, header=None, names=column_names)

# Plot distribution of Diagnosis
plt.figure(figsize=(8, 6))
sns.countplot(df['Diagnosis'])
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.title('Distribution of Diagnosis')
plt.show()

# Encode the categorical data values
labelencoder_Y = LabelEncoder()
df['Diagnosis'] = labelencoder_Y.fit_transform(df['Diagnosis'])

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.iloc[:, 1:12].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show() 

# Pairplot including 'Diagnosis'
sns.pairplot(df.iloc[:, 1:12], hue='Diagnosis')
plt.show()

# Display the loaded dataset
print(df.head(5))
print(df)

# Split the dataset's Independent (X) & Dependent (Y)
X = df.iloc[:, 2:31].values
Y = df.iloc[:, 1].values

# Split the datasets into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Function for models
def models(X_train, Y_train):
    # Logistic regression
    log = LogisticRegression(random_state=0)
    log.fit(X_train, Y_train)
    
    # Decision Tree
    tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    tree.fit(X_train, Y_train)
    
    # Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    rf.fit(X_train, Y_train)
    
    # Support Vector Machine
    svm = SVC(kernel='linear', random_state=0)
    svm.fit(X_train, Y_train)
    
    return log, tree, rf, svm

# Get the models
log, tree, rf, svm = models(X_train, Y_train)

# Print the models' accuracies
models_list = [log, tree, rf, svm]
for i, model in enumerate(models_list):
    print(f'Model {i} Training Accuracy: {model.score(X_train, Y_train)}')

# Test models accuracy on test data
for i, model in enumerate(models_list):
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f'Model {i} Testing Accuracy: {accuracy}')
    cm = confusion_matrix(Y_test, Y_pred)
    print(f'Confusion Matrix for Model {i}:\n{cm}')

# Test metrics of the model
for i, model in enumerate(models_list):
    print("Model", i)
    print(classification_report(Y_test, model.predict(X_test)))
    print(accuracy_score(Y_test, model.predict(X_test)))
    print()

# Print the predictions of Random Forest model
pr = models_list[2].predict(X_test)
print(pr)
print()
print(Y_test)
################################################################ 
#NN Pytorch model

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

# Create DataLoader for training and testing
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define the neural network 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(29, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Output layer with 2 classes

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the neural network
model = NeuralNetwork()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the neural network
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# Evaluate the neural network
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Accuracy on test set: {100 * correct / total}%")
