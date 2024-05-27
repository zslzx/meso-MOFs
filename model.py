import torch
import torch.nn as nn
import torch.nn.functional as F

class CLSModel(nn.Module):
    def __init__(self):
        super(CLSModel,self).__init__()

        self.layer1 = nn.Linear(5, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 2)
    
    def forward(self, x, input_dim=5):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        cls = F.softmax(x, dim=-1)[:,-1]
        return cls, x

class RGRModel(nn.Module):
    def __init__(self):
        super(RGRModel,self).__init__()
        self.layer1 = nn.Linear(5, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 1)
    
    def forward(self, x, input_dim=5):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x 