import torch

class LeastSquares(torch.nn.Module):
    def __init__(self, num_parameters):
        super(LeastSquares, self).__init__()
        self.alphas = torch.rand((num_parameters+2, 1), requires_grad=True)
    
    def forward(self, x):
        x = torch.cat((x, x[:,0:1]**2, x[:,0:1]**3), dim=1)
        return x@self.alphas.squeeze()