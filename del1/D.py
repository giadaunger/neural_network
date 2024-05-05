import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform and load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Define the neural network architecture
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
in_features = 28 * 28  # 784, as images are 28x28 pixels
h1 = 128
h2 = 64
out_features = 10

# Initialize the model, optimizer, and loss function
model = Perceptron(in_features, h1, h2, out_features).to(device)
optimizer = Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Train the model
epochs = 10
train_losses = []
train_accuracies = []
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
    train_losses.append(total_loss / len(train_loader))
    train_accuracies.append(100 * correct / total)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracies[-1]:.2f}%')

# Plotting training metrics
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, marker='o', linestyle='-', color='blue')
plt.title('Training Loss Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_accuracies, marker='o', linestyle='-', color='red')
plt.title('Training Accuracy Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.show()

# Evaluate the model
model.eval()
correct = 0
total = 0
misclassified_samples = []
with torch.no_grad():
    for data, targets in test_loader:
        data = data.view(data.size(0), -1).to(device)
        targets = targets.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        # Collect misclassified samples
        mask = predicted != targets
        for img, true_label, pred_label in zip(data[mask], targets[mask], predicted[mask]):
            if len(misclassified_samples) < 10:  # Only save 10 samples to plot
                misclassified_samples.append((img, true_label.item(), pred_label.item()))

accuracy = 100 * correct / total
print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')

# Plotting function for misclassified samples
def plot_misclassified(samples):
    plt.figure(figsize=(10, 5))
    for idx, (image, true, pred) in enumerate(samples):
        image = image.view(28, 28).cpu().numpy()
        plt.subplot(2, 5, idx + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'True: {true}\nPred: {pred}')
        plt.axis('off')
    plt.suptitle('Misclassified Images')
    plt.show()

# Plot misclassified images
plot_misclassified(misclassified_samples)
