import torch
import torch.nn as nn
import torch.optim as optim

class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 7)  # Output for emotion classification (7 classes)
        
        self.fc3 = nn.Linear(512, 2)  # Output for valence-arousal regression

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)

        x = x.view(x.size(0), -1)  # Flatten
        
        x = torch.relu(self.fc1(x))
        
        emotion_class_output = self.fc2(x)  # Emotion class prediction
        valence_arousal_output = self.fc3(x)  # Valence-arousal prediction
        
        return emotion_class_output