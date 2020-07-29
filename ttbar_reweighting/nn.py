import torch.nn as nn
import torch.nn.functional as F

class ReweightingNet(nn.Module):
    def __init__(self, num_inputs):
        super(ReweightingNet, self).__init__()

        self.fc1 = nn.Linear(num_inputs, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, 32)
        self.fc6 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = F.leaky_relu(self.fc3(x), 0.1)
        x = F.leaky_relu(self.fc4(x), 0.1)
        x = F.leaky_relu(self.fc5(x), 0.1)
        return self.fc6(x)
