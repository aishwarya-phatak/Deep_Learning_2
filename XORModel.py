import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.hidden = nn.Linear(2,4)
        self.output = nn.Linear(4,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x
