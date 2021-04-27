import torch
import torch.nn as nn

class ZiNet(nn.Module):
    def __init__(self):
        super(ZiNet,self).__init__()
        
        self.layer_1 = nn.Linear(784,128)
        self.activate = nn.ReLU()
        self.layer_2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.layer_1(x)
        x = self.activate(x)
        x = self.layer_2(x)

        return x