def eval_model(model: torch.nn,Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn = loss_fn,
               accuracy_fn = accuracy_fn,
               device = device):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            #Send data to the GPU
            X, y = X.to(device), y.to(device)

            # 1.Forward Pass
            y_pred = Model(X)

            # 2. Calculate loss and accuracy
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true = y,
                               y_pred = y_pred.argmax(dim = 1)) # Logits -> Labels)
            
            # Scale the loss and acc by the number of batches
            loss /= len(data_loader)
            acc /= len(data_loader)
    
    return{"model_name": model.class_name__,
           "model_loss": loss.item(),
           "model_acc": acc}
