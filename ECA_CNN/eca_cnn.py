
import torch
import torch.nn as nn
import torch.nn.functional as F
from eca_module import ECALayer

class ECACNN(nn.Module):
    def __init__(self):
        super(ECACNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.eca1 = ECALayer(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.eca2 = ECALayer(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.eca3 = ECALayer(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.eca1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.eca2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(F.relu(self.eca3(self.conv3(x))))
        x = x.view(-1, 256 * 32 * 32)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x