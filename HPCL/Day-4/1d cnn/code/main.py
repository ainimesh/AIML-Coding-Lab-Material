import numpy as np
import torch
from data import DataModule
from model import CNN1D
from tqdm import tqdm
import random


if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    # torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_module = DataModule(batch_size=32, window_size=50)
    data_module.prepare_data()
    data_module.setup()
    
    train_dataloader = data_module.train_dataloader()
    
    model = CNN1D().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Set the number of epochs
    epochs = 1000
    
    # TRAIN
    losses = list()
    for epoch in tqdm(range(epochs)):
        model.train()
        loss_epoch = 0
        print("Epoch:")
        for x, y in tqdm(train_dataloader):
            x, y = x.to(device), y.long().to(device)
            # Clear all previous gradients
            optimizer.zero_grad()
            # Generate predictions
            preds = model.forward(x)
            # Calculate the loss
            loss = criterion(preds, y)
            # Calculate the gradients
            loss.backward()
            # Finally, update the weights
            optimizer.step()
            loss_epoch+= loss.item()
        losses.append(loss_epoch/len(train_dataloader))
        print("Loss:", losses[-1])
        print("\n")