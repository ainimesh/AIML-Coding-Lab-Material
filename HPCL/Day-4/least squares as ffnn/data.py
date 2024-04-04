import numpy as np
import torch.utils
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.utils.data



class DataModule():
    def __init__(self, num_points, batch_size):
        self.num_points = num_points
        self.batch_size = batch_size
    
    def prepare_data(self):
        # Generate Training Data
        self.X_train = np.random.uniform(low=-1, high=1, size=(self.num_points, 1))
        # Add a feature which will act as bias
        self.X_train = np.concatenate((self.X_train, np.ones((self.num_points, 1))), axis=1)
        # Generate the target variable
        self.Y_train = ((1.3*self.X_train[:,0] + 0.8*self.X_train[:,0]**2 + 3.6*self.X_train[:,0]**3 + 10*self.X_train[:,1]) + np.random.uniform(low=-0.4, high=0.4, size=self.num_points))
        
        # Generate Test Data
        self.X_test = np.random.uniform(low=-2, high=2, size=(self.num_points, 1))
        # Add a feature which will act as bias
        self.X_test = np.concatenate((self.X_test, np.ones((self.num_points, 1))), axis=1)
        # Generate the target variable
        self.Y_test = ((1.3*self.X_test[:,0] + 0.8*self.X_test[:,0]**2 + 3.6*self.X_test[:,0]**3 + 10*self.X_test[:,1]) + np.random.uniform(low=-0.4, high=0.4, size=self.num_points))
    
    def setup(self):
        self.train_dataset = TensorDataset(torch.Tensor(self.X_train), torch.Tensor(self.Y_train))
        self.test_dataset = TensorDataset(torch.Tensor(self.X_test), torch.Tensor(self.Y_test))
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)