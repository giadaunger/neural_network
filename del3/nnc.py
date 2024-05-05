import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Define transformations for the input data
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Download and load the training and test datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load a pre-trained ResNet50 model
resnet50 = models.resnet50(pretrained=True)

# Modify the classifier for CIFAR-10
num_ftrs = resnet50.fc.in_features
resnet50.fc = torch.nn.Linear(num_ftrs, 10)

# Move the model to the device
resnet50 = resnet50.to(device)

# Define loss function
criterion = torch.nn.CrossEntropyLoss()

# Define optimizer (with lr set and weight decay for regularization)
optimizer = optim.Adam(resnet50.parameters(), lr=0.001, weight_decay=0.0001)

# Function to train the model and record loss
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Set model to training mode
    train_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        total = 0
        correct = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = 100 * correct / total
        train_losses.append((epoch_loss, epoch_accuracy))
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
        
    print('Training complete')
    return train_losses

# Train the model
train_losses = train_model(resnet50, train_loader, criterion, optimizer, num_epochs=10)

# Function to evaluate the model and capture misclassified images
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluate mode
    correct = 0
    total = 0
    misclassified_examples = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Capture misclassified images
            misclassified = predicted != labels
            for image, label, pred in zip(inputs[misclassified], labels[misclassified], predicted[misclassified]):
                misclassified_examples.append((image, label, pred))

            if len(misclassified_examples) > 10:  # Limit to some number to plot later
                break

    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {100 * correct / total:.2f}%')
    return misclassified_examples, accuracy

# Evaluate the model
misclassified_examples, accuracy = evaluate_model(resnet50, test_loader)

# Plotting training loss and accuracy
epochs = range(1, len(train_losses) + 1)
losses = [x[0] for x in train_losses]
accuracies = [x[1] for x in train_losses]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, losses, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracies, label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting misclassified images
def plot_misclassified(misclassified_examples):
    plt.figure(figsize=(15, 4))
    for i, (image, label, pred) in enumerate(misclassified_examples[:10]):
        image = image.cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        plt.subplot(2, 5, i + 1)
        plt.imshow(image)
        plt.title(f'True: {label.item()}\nPred: {pred.item()}')
        plt.axis('off')
    plt.show()

plot_misclassified(misclassified_examples)
