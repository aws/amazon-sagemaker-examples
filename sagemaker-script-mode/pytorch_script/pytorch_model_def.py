import torch
import torch.nn as nn

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(13, 13)
        self.fc2 = nn.Linear(13, 6)
        self.fc3 = nn.Linear(6, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


def get_model():
    
    model = NeuralNet()
    return model
