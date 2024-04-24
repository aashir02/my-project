import torch
import torch.nn as nn
import torchS
import torch.nn as nn
import torchvision.models as models

class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        self.resnet18 = models.resnet18(weights=models.resnet.ResNet18_Weights.DEFAULT)
        # Freeze all the parameters in the pre-trained model
        for param in self.resnet18.parameters():
            param.requires_grad = False
        # Modify the final fully connected layer
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)


# Usage example:
model = CustomResNet18(num_classes=2)  # Change num_classes based on your task


