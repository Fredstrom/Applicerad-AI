import torch.cuda
import torch.nn as nn


class CNN_model(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_model, self).__init__()

        # Layer 1
        # Image shape: (batch_size, 3, image_size)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)

        # Image shape: (batch_size, 12, image_size)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        # Max pooling with kernel size 2 results in half the size of the original image.
        # Image shape: (batch_size, 12, image_size/2)

        # Layer 2
        # Image shape: (batch_size, 3, image_size)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)

        # Image shape: (batch_size, 24, image_size)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)
        # Image shape: (batch_size, 24, image_size/4)

        # Layer 3
        # Image shape: (batch_size, 3, image_size)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Image shape: (batch_size, 32, image_size)
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(kernel_size=2)
        # Image shape: (batch_size, 32, image_size/8)

        self.fc = nn.Linear(in_features= 50 * 50 * 32, out_features=40000)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(in_features= 40000, out_features= num_classes)

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

        x = x.view(-1, 32 * 50 * 50)

        x = self.fc(x)
        x = self.relu_fc(x)
        x = self.fc2(x)

        return x
