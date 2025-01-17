import torch
from torchvision import models

# Define the neural network model
class EmotionClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Using a pre-trained ResNet18
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)