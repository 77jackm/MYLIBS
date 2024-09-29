## BREN This library contains a set of Python methods for training and
## testing a pytorch model. They require torch and torch.utils.data
## to be installed.

import torch
from torch import nn

def train_loop(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device = "None"):
    """
    Train loop for a given model, data loader, loss function, optimizer, and accuracy function.

    This function trains the model on the given data loader and calculates the train loss and accuracy.

    Args:
        model (torch.nn.Module): The model to train
        data_loader (torch.utils.data.DataLoader): The data loader to use
        loss_fn (torch.nn.Module): The loss function to use
        optimizer (torch.optim.Optimizer): The optimizer to use
        accuracy_fn: The accuracy function to use
        device (torch.device, optional): The device to use. Defaults to the device the model is on.

    Returns:
        None
    """
    train_loss, train_acc = 0, 0
    model.to(device)
    for batch, (X, y) in enumerate(data_loader):
        # Send data to GPU
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y_true=y,
                                 y_pred=y_pred.argmax(dim=1)) # Go from logits -> pred labels

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

    # Calculate loss and accuracy per epoch and print out what's happening
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")



def test_loop(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device = "None"):
    """
    Test loop for a given model, data loader, loss function, and accuracy function.

    This function evaluates the model on the given test data loader and calculates the test loss and accuracy.

    Args:
        data_loader (torch.utils.data.DataLoader): The test data loader
        model (torch.nn.Module): The model to evaluate
        loss_fn (torch.nn.Module): The loss function to use
        accuracy_fn: The accuracy function to use
        device (torch.device, optional): The device to use. Defaults to the device the model is on.

    Returns:
        None
    """
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval() # put model in eval mode
    # Turn on inference context manager
    with torch.inference_mode(): 
        for X, y in data_loader:
            # Send data to GPU
            X, y = X.to(device), y.to(device)
            
            # 1. Forward pass
            test_pred = model(X)
            
            # 2. Calculate loss and accuracy
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y,
                y_pred=test_pred.argmax(dim=1) # Go from logits -> pred labels
            )
        
        # Adjust metrics and print out
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


def eval_model(data_loader: torch.utils.data.DataLoader,
                model: torch.nn.Module,
                loss_fn: torch.nn.Module,
                accuracy_fn,
                device = "None"):
    """
    Evaluate a model on a given data loader.

    Args:
        model (torch.nn.Module): The model to evaluate
        data_loader (torch.utils.data.DataLoader): The data loader to use
        loss_fn (torch.nn.Module, optional): The loss function to use. Defaults to torch.nn.Module.
        accuracy_fn: The accuracy function to use
        device (torch.device, optional): The device to use. Defaults to the device the model is on.

    Returns:
        A dictionary containing the model name, loss, and accuracy.
    """
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            #Send data to the GPU
            X, y = X.to(device), y.to(device)

            # 1.Forward Pass
            y_pred = model(X)

            # 2. Calculate loss and accuracy
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true = y,
                               y_pred = y_pred.argmax(dim = 1)) # Logits -> Labels)
            
            # Scale the loss and acc by the number of batches
            loss /= len(data_loader)
            acc /= len(data_loader)
    
    return{"model_name": model.__class__.__name__,
           "model_loss": loss.item(),
           "model_acc": acc}