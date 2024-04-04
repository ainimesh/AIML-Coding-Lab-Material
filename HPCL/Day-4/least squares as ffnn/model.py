import torch

class LeastSquares(torch.nn.Module):
    def __init__(self, num_parameters):
        super(LeastSquares, self).__init__()
        self.layer1 = torch.nn.Linear(num_parameters, 1024, bias=False)
        self.layer2 = torch.nn.Linear(1024, 1)
    
    def forward(self, x):
        # x = torch.cat((x, x[:,0:1]**2, x[:,0:1]**3), dim=1)
        x = torch.nn.Tanh()(self.layer1(x))
        out = self.layer2(x)
        return out.squeeze()