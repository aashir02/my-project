import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
DATADIR=r"C:\Users\ashir\OneDrive\Pictures\MyDataset\Train"
#reading dataset
from PIL import Image

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesses an image for ResNet-18 input.
    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired image size (default: (224, 224)).
    Returns:
        PIL.Image: Preprocessed image as a PIL image.
    """
    # Load the image
    image = Image.open(image_path)
    # Resize to target size
    image = image.resize(target_size)
    return image



my_batch_size=30
#dataset augmentation
def get_augmentation_transforms():
    """
    Returns a composition of data augmentation transformations.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])


import os
import shutil
from torch.utils.data import Dataset, DataLoader
class DeepfakeDataset(Dataset):
    def __init__(self, DATADIR, transform=None):
        self.DATADIR = DATADIR
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load paths for real images
        real_dir = os.path.join(DATADIR, "real")
        real_images = os.listdir(real_dir)
        self.image_paths.extend([os.path.join(real_dir, img) for img in real_images])
        self.labels.extend([0] * len(real_images))  # Assign label 0 for real images

        # Load paths for fake images
        fake_dir = os.path.join(DATADIR, "fake")
        fake_images = os.listdir(fake_dir)
        self.image_paths.extend([os.path.join(fake_dir, img) for img in fake_images])
        self.labels.extend([1] * len(fake_images))  # Assign label 1 for fake images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = preprocess_image(image_path)
        if self.transform:
            augmented_image = self.transform(image)  # Apply data augmentation
        else:
            augmented_image = image
        label = self.labels[idx]
        return augmented_image, label
    

    def split_dataset(self, train_ratio, val_ratio, test_ratio, seed):
        # Shuffling the dataset indices
        indices = np.arange(len(self))
        np.random.seed(seed)
        np.random.shuffle(indices)

        # Calculate the number of samples for each split
        num_samples = len(self)
        num_train = int(train_ratio * num_samples)
        num_val = int(val_ratio * num_samples)
        num_test = num_samples - num_train - num_val

        # Assign samples to each split
        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]

        # Define lists to hold image paths and labels for each split
        train_image_paths = [self.image_paths[i] for i in train_indices]
        val_image_paths = [self.image_paths[i] for i in val_indices]
        test_image_paths = [self.image_paths[i] for i in test_indices]

        train_labels = [self.labels[i] for i in train_indices]
        val_labels = [self.labels[i] for i in val_indices]
        test_labels = [self.labels[i] for i in test_indices]

        # Create directories for train, val, and test sets
        train_dir = os.path.join(self.DATADIR, 'train')
        val_dir = os.path.join(self.DATADIR, 'val')
        test_dir = os.path.join(self.DATADIR, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Save the split information
        self.train_image_paths = train_image_paths
        self.val_image_paths = val_image_paths
        self.test_image_paths = test_image_paths

        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
    
    def copy_images_to_directory(self, image_paths, destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            destination = os.path.join(destination_dir, image_name)
            shutil.copy(image_path, destination)
import torch.nn as nn
# Example usage:
# Define your dataset directory
DATADIR=r"C:\Users\ashir\OneDrive\Pictures\MyDataset\Train"

# Create an instance of DeepfakeDataset without specifying the split argument
dataset = DeepfakeDataset(DATADIR)

# Split the dataset into train, validation, and test sets
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
seed = 42  # Set a seed for reproducibility
dataset.split_dataset(train_ratio, val_ratio, test_ratio, seed)

dataset.copy_images_to_directory(dataset.train_image_paths, os.path.join(DATADIR, 'train'))
dataset.copy_images_to_directory(dataset.val_image_paths, os.path.join(DATADIR, 'val'))
dataset.copy_images_to_directory(dataset.test_image_paths, os.path.join(DATADIR, 'test'))


#model
from resnet_model import CustomResNet18
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define your dataset and data loader
transform = get_augmentation_transforms()  # Define your preprocessing and augmentation transformations
train_dataset = DeepfakeDataset(DATADIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=my_batch_size, shuffle=True)
val_dataset = DeepfakeDataset(DATADIR, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=my_batch_size, shuffle=False)

# Initialize the pre-trained ResNet-18 model
model = CustomResNet18(num_classes=1)  # No need to specify num_classes for binary classification

# Define your loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Move the model to the appropriate device (e.g., GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


#validation
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, val_loader, device):
    """
    Evaluate the model's performance on the validation set.

    Args:
        model: The trained neural network model.
        val_loader: DataLoader for the validation dataset.
        device: Device to run the evaluation on (e.g., 'cuda' or 'cpu').

    Returns:
        accuracy (float): Accuracy of the model on the validation set.
        precision (float): Precision of the model on the validation set.
        recall (float): Recall of the model on the validation set.
        f1 (float): F1-score of the model on the validation set.
    """
    model.eval()  # Set the model to evaluation mode
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            predictions = torch.round(torch.sigmoid(outputs))

            val_predictions.extend(predictions.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())

    # Calculate evaluation metrics
    val_accuracy = accuracy_score(val_targets, val_predictions)
    val_precision = precision_score(val_targets, val_predictions)
    val_recall = recall_score(val_targets, val_predictions)
    val_f1 = f1_score(val_targets, val_predictions)

    return val_accuracy, val_precision, val_recall, val_f1

    # After each epoch during training, call the evaluate_model function
    for epoch in range(num_epochs):
        # Training loop...

        # Evaluate the model on the validation set
        val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, device)
        if val_accuracy is not None and val_precision is not None and val_recall is not None and val_f1 is not None:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}, '
                f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}')
        else:
            print("Error: Evaluation metrics are not available.")

    print("Training complete")
    

# Define the function to train the model with visualization
def train_model_with_visualization(model, train_loader, criterion, optimizer, num_epochs, device):
    """
    Train the model with visualization of training metrics.

    Args:
        model: The neural network model to be trained.
        train_loader: DataLoader for the training dataset.
        criterion: The loss function.
        optimizer: The optimizer used for training.
        num_epochs (int): Number of epochs for training.
        device: Device to run the training on (e.g., 'cuda' or 'cpu').

    Returns:
        None
    """
    # Lists to store training metrics
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            # Move inputs and labels to the device
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track the loss and accuracy
            running_loss += loss.item() * inputs.size(0)
            predictions = torch.round(torch.sigmoid(outputs))
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        # Calculate epoch-level training statistics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / total_samples
        
        # Append training metrics to lists for visualization
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, device)
        # Print epoch-level training statistics
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}, 'f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}')

    # Plot training metrics
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("Training complete")

# Train the model with visualization
num_epochs = 10  # Adjust as needed
train_model_with_visualization(model, train_loader, criterion, optimizer, num_epochs, device)
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import itertools
def hyperparameter_tuning(train_loader, val_loader, criterion, device):
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [16, 32, 64]
    num_epochs = [10, 20, 30]
    optimizers = ['adam', 'sgd', 'rmsprop']

    best_f1_score = 0
    best_hyperparameters = {}

    for lr, batch_size, epochs, optimizer_name in itertools.product(learning_rates, batch_sizes, num_epochs, optimizers):
        # Initialize the model with current hyperparameters
        model = CustomResNet18(num_classes=1)
        model.to(device)

        # Define optimizer based on the current optimizer_name
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        elif optimizer_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

        train_losses = []
        train_accuracies = []

        for epoch in range(epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for inputs, labels in train_loader:
                # Move inputs and labels to the device
                inputs = inputs.to(device)
                labels = labels.float().to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Track the loss and accuracy
                running_loss += loss.item() * inputs.size(0)
                predictions = torch.round(torch.sigmoid(outputs))
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
            
            # Calculate epoch-level training statistics
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_accuracy = correct_predictions / total_samples

            # Append training metrics to lists for visualization
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)

        # Evaluate the model on the validation set
        val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, device)

        # Check if the current F1 score is better than the best so far
        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            best_hyperparameters = {'learning_rate': lr, 'batch_size': batch_size, 'epochs': epochs, 'optimizer': optimizer_name}

    return best_hyperparameters


# Perform hyperparameter tuning
best_hyperparameters = hyperparameter_tuning(train_loader, val_loader, criterion, device)
print("Best hyperparameters:", best_hyperparameters)


