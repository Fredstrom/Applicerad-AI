import torch.cuda
import torch.nn as nn
from settings import settings


class CNN_model(nn.Module):
    def __init__(self, num_classes=2):
        self.width = settings["image_size"][0]
        self.height = settings["image_size"][1]
        super(CNN_model, self).__init__()

        # Layer 1
        # Image shape: (batch_size, 3, image_size)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)

        # Image shape: (batch_size, 12, image_size)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        # Max pooling with kernel size 2 results in half the size of the original image.

        # Layer 2
        # Image shape: (batch_size, 12, image_size / 2)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)

        # Image shape: (batch_size, 24, image_size / 2)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)

        # Layer 3
        # Image shape: (batch_size, 24, image_size / 4)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=6, kernel_size=3, stride=1, padding=1)

        # Image shape: (batch_size, 6, image_size / 4)
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(kernel_size=2)

        # Image shape: (batch_size, 6, image_size / 8)
        self.fc = nn.Linear(in_features= int(self.width / 8) * int(self.height / 8) * 6, out_features=2)
#

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.max_pool3(x)

        x = x.view(-1, 6 * int(self.height / 8) * int(self.width / 8))

        x = self.fc(x)

        return x

