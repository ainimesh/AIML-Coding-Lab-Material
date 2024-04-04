import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from data import DataModule
import torch
from tqdm import tqdm
from model import LeastSquares
from utility import generate_scatter_plot, generate_line_plot, generate_multiple_scatter_plots
import random



if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    # Generate data
    data_module = DataModule(num_points=200, batch_size=64)
    data_module.prepare_data()
    data_module.setup()
    # Check the shape of the data
    print(data_module.X_train.shape, data_module.Y_train.shape)
    # Let's check how our data looks
    generate_scatter_plot(data_module.X_train[:,0], data_module.Y_train, "ground_truth_train")
    
    # Load the dataloader
    train_dataloader = data_module.train_dataloader()
    # Create a model
    model = LeastSquares(data_module.X_train.shape[-1])
    # Choose an optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # Set the number of epochs
    epochs = 1000
    
    # TRAIN
    losses = list()
    for epoch in tqdm(range(epochs)):
        model.train()
        loss_epoch = 0
        for x, y in train_dataloader:
            # Clear all previous gradients
            optimizer.zero_grad()
            # Generate predictions
            preds = model.forward(x)
            # Calculate the loss
            loss = torch.mean(torch.square(preds-y))
            # Calculate the gradients
            loss.backward()
            # Finally, update the weights
            optimizer.step()
            loss_epoch+= loss.item()
        losses.append(loss_epoch/len(train_dataloader))
    # Check out the loss curves
    generate_line_plot(np.arange(1, epochs+1), losses, "loss")
    
    
    # TEST
    model.eval()
    # We will save all our predictions in this list
    final_preds = list()
    test_dataloader = data_module.test_dataloader()
    # Get all tbe predictions
    for x, y in test_dataloader:
        preds = model.forward(x)
        final_preds.extend(preds.detach().numpy())
    final_preds = np.asarray(final_preds)
    print("\nLoss on test set", np.mean(np.square(final_preds-data_module.Y_test)))
    # Plot it out
    generate_multiple_scatter_plots(data_module.X_test[:,0], data_module.Y_test, final_preds, "predicted_test")