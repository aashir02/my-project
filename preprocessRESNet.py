import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from resnet_model import CustomResNet18
lr_rate=0.01
my_batch_size=64
num_epochs = 30
optimizer_name ='rmsprop'
target_size = (224, 224)

class DeepfakeDataset(Dataset):
    def __init__(self, DATADIR, transform=None,target_size=(224, 224)):
        self.DATADIR = DATADIR
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.target_size = target_size

        real_dir = os.path.join(DATADIR, "real")
        real_images = os.listdir(real_dir)
        self.image_paths.extend([os.path.join(real_dir, img) for img in real_images])
        self.labels.extend([0] * len(real_images))  

        fake_dir = os.path.join(DATADIR, "fake")
        fake_images = os.listdir(fake_dir)
        self.image_paths.extend([os.path.join(fake_dir, img) for img in fake_images])
        self.labels.extend([1] * len(fake_images))  

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image = image.resize(self.target_size)
        image = transforms.ToTensor()(image)  # Convert PIL image to tensor
        label = self.labels[idx]
        return image, label



    
    def split_dataset(self, train_ratio, val_ratio, test_ratio, seed):
        indices = np.arange(len(self))
        np.random.seed(seed)
        np.random.shuffle(indices)

        num_samples = len(self)
        num_train = int(train_ratio * num_samples)
        num_val = int(val_ratio * num_samples)

        train_indices = indices[:num_train]
        val_indices = indices[num_train:num_train + num_val]
        test_indices = indices[num_train + num_val:]

        self.train_image_paths = [self.image_paths[i] for i in train_indices]
        self.val_image_paths = [self.image_paths[i] for i in val_indices]
        self.test_image_paths = [self.image_paths[i] for i in test_indices]

        self.train_labels = [self.labels[i] for i in train_indices]
        self.val_labels = [self.labels[i] for i in val_indices]
        self.test_labels = [self.labels[i] for i in test_indices]
    
    def copy_images_to_directory(self, image_paths, destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            destination = os.path.join(destination_dir, image_name)
            shutil.copy(image_path, destination)

def get_augmentation_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])

def preprocess_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path)
    image = image.resize(target_size)
    return image

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            predictions = torch.round(torch.sigmoid(outputs))
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / total_samples
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {val_accuracy:.4f}, 'f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}')
        # After training your model
        torch.save(model.state_dict(), 'model.pth')
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

def evaluate_model(model, test_loader, device):
    model.eval()
    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            predictions = torch.round(torch.sigmoid(outputs))

            test_predictions.extend(predictions.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    test_accuracy = accuracy_score(test_targets, test_predictions)
    test_precision = precision_score(test_targets, test_predictions,zero_division=1)
    test_recall = recall_score(test_targets, test_predictions,zero_division=1)
    test_f1 = f1_score(test_targets, test_predictions)

    return test_accuracy, test_precision, test_recall, test_f1

def visualize_predictions(model, test_loader, device, num_images=5):
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            predictions = torch.round(torch.sigmoid(outputs))

            # Convert tensors to numpy arrays
            inputs_np = inputs.cpu().numpy()
            labels_np = labels.cpu().numpy()
            predictions_np = predictions.cpu().numpy()

            # Visualize a fixed number of images in a horizontal fashion
            for i in range(min(num_images, inputs_np.shape[0])):
                image = inputs_np[i].transpose(1, 2, 0)
                true_label = labels_np[i]
                predicted_label = predictions_np[i]

                plt.subplot(1, num_images, i + 1)
                plt.imshow(image)
                plt.title(f'True: {true_label}, Predicted: {predicted_label}')
                plt.axis('off')

            plt.show()
            break  # Only visualize one batch of images




def main():
    DATADIR = r"C:\Users\ashir\OneDrive\Pictures\MyDataset\Train"
    transform = get_augmentation_transforms()

    dataset = DeepfakeDataset(DATADIR, transform=transform, target_size=target_size)
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    seed = 42
    dataset.split_dataset(train_ratio, val_ratio, test_ratio, seed)

    dataset.copy_images_to_directory(dataset.train_image_paths, os.path.join(DATADIR, 'train'))
    dataset.copy_images_to_directory(dataset.val_image_paths, os.path.join(DATADIR, 'val'))
    dataset.copy_images_to_directory(dataset.test_image_paths, os.path.join(DATADIR, 'test'))

    train_loader = DataLoader(dataset, batch_size=my_batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=my_batch_size, shuffle=False)

    model = CustomResNet18(num_classes=1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
   

    test_dataset = DeepfakeDataset(DATADIR)
    test_loader = DataLoader(test_dataset, batch_size=my_batch_size, shuffle=False)
    """
    test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test F1-score: {test_f1:.4f}')
    """
    # Assuming you have already defined your model, test_loader, and device
    #visualize_predictions(model, test_loader, device, num_images=5)

if __name__ == "__main__":
    main()
