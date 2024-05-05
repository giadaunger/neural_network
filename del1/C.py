import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Transforms for the input data
transform = transforms.Compose([ 
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)) 
])

# Load MNIST dataset
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Data loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Perceptron class
class Perceptron(nn.Module):
    def __init__(self, in_features, h1, h2, out_features):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Model parameters
in_features = 28 * 28  # 784 input features (28x28 pixels)
h1 = 128               # First hidden layer
h2 = 64                # Second hidden layer
out_features = 10      # 10 classes for MNIST digits

# Initialize model, optimizer, and loss function
perceptron = Perceptron(in_features, h1, h2, out_features)
optimizer = Adam(perceptron.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training the model
epochs = 10
loss_history = []
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.view(data.size(0), -1)
        outputs = perceptron(data)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

# Plot the training loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), loss_history, marker='o', linestyle='-', color='b')
plt.title('Training Loss Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.xticks(range(1, epochs+1))
plt.show()

# Evaluate the model
perceptron.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, targets in test_loader:
        data = data.view(data.size(0), -1)
        outputs = perceptron(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')