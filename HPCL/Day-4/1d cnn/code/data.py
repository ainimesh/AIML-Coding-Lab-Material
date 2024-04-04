import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader



class DataModule():
    def __init__(self, batch_size, window_size, test_size=0.2):
        self.batch_size = batch_size
        self.window_size = window_size
        self.test_size = test_size
    
    def prepare_data(self):
        X = list()
        Y = list()
        for i in range(1,16):
            data = np.loadtxt("data/"+str(i)+".csv", delimiter=",", dtype=float)
            X.extend(data[:,1:-1])
            Y.extend(data[:,-1])
        X = np.asarray(X)
        Y = np.asarray(Y, dtype=np.int32)
        
        # Create a train-test split
        num_train_samples = int((1-self.test_size)*len(X))
        X_train, X_test = X[:num_train_samples], X[num_train_samples:]
        Y_train, Y_test = Y[:num_train_samples]-1, Y[num_train_samples:]-1
        
        # Now generate rolling window
        self.X_train, self.Y_train = list(), list()
        for label in [1,2,3,4,5,6,7]:
            indices = np.where(Y_train==label)[0]
            for i in tqdm(range(len(indices)-self.window_size)):
                self.X_train.append(X_train[indices[i:i+self.window_size]])
                self.Y_train.append(label)
        
        self.X_test, self.Y_test = list(), list()
        for label in [1,2,3,4,5,6,7]:
            indices = np.where(Y_test==label)[0]
            for i in tqdm(range(len(indices)-self.window_size)):
                self.X_test.append(X_test[indices[i:i+self.window_size]])
                self.Y_test.append(label)
        
        self.X_train = np.asarray(self.X_train)
        self.Y_train = np.asarray(self.Y_train)
        self.X_test = np.asarray(self.X_test)
        self.Y_test = np.asarray(self.Y_test)
        
        print(self.X_train.shape, self.Y_train.shape)
        print(self.X_test.shape, self.Y_test.shape)
        
    def setup(self):
        self.train_dataset = TensorDataset(torch.Tensor(self.X_train), torch.Tensor(self.Y_train))
        self.test_dataset = TensorDataset(torch.Tensor(self.X_test), torch.Tensor(self.Y_test))\
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)