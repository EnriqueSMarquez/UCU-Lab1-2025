from torch import nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, nb_classes, in_channels, img_size=(255, 255)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=5, padding=2)
        self.max_pool = nn.MaxPool2d((2, 2), stride=2)
        vector_size = img_size // 2 * img_size // 2 * 64
        self.fc = nn.Linear(vector_size, nb_classes)
        self.nb_classes = nb_classes
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(len(x), -1)
        output = self.fc(x)
        return output