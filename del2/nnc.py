import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations for training
train_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2)
])

# Transformations for testing (no data augmentation)
test_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Define the neural network architecture with regularization
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch normalization layer
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)  # Dropout layer
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Initialize the model, optimizer, and loss function with weight decay
model = CNN().to(device)
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)  # Added weight decay
loss_fn = nn.CrossEntropyLoss()

# Directory for saving models
save_path = './model_checkpoints'
os.makedirs(save_path, exist_ok=True)

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
        data = data.to(device)
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

    # Save the model's state dictionary
    torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch+1}.pth'))
    print(f'Model saved to {os.path.join(save_path, f"epoch_{epoch+1}.pth")}')

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
        data = data.to(device)
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
        image = image.squeeze().cpu().numpy()
        plt.subplot(2, 5, idx + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'True: {true}\nPred: {pred}')
        plt.axis('off')
    plt.suptitle('Misclassified Images')
    plt.show()

# Plot misclassified images
plot_misclassified(misclassified_samples)
