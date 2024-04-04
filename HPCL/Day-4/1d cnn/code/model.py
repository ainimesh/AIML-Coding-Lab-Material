import torch


class CNN1D(torch.nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.layer1 = torch.nn.Conv1d(3, 64, kernel_size=6)
        self.layer2 = torch.nn.Conv1d(64, 64, kernel_size=6)
        self.layer3 = torch.nn.Linear(64, 7)
    
    def forward(self, x):
        x = torch.swapdims(x, 2, 1)
        x = torch.nn.ReLU()(self.layer1(x))
        x = torch.nn.ReLU()(self.layer2(x))
        x = torch.mean(x, dim=2)
        out = self.layer3(x)
        return out