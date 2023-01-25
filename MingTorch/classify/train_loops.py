import os,sys
sys.path.append(os.getcwd())

import torch
import numpy as np
from tqdm import tqdm

torch.backends.cudnn.enable =True

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.sampler)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    
    pbar = tqdm(total=size)
    for batch, (X, y) in enumerate(dataloader):
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        pbar.set_description(f"[train] loss: {loss:>7f}  [{batch+1:>3d}/{num_batches:>3d}]")
        pbar.update(len(X))
    
    train_loss /= num_batches
    correct /= size
    return {
        "train_Avg_loss":train_loss,
        "train_correct": correct,
    }

def val_loop(dataloader, model, loss_fn=None):
    model.eval()
    size = len(dataloader.sampler)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    pabr = tqdm(total=size)
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            desc = "[val] acc: {:7f} [{:>3d}/{:3d}]".format(correct/len(X), batch+1, num_batches)
            if loss_fn is None:
                loss = loss_fn(pred, y)
                test_loss += loss.item()
                desc = f"[val] loss: {loss:>7f}  [{batch+1:>3d}/{num_batches:>3d}]"
            
            pabr.set_description(desc)
            pabr.update(len(X))
            
    test_loss /= num_batches
    correct /= size
    return {
        "val_Avg_loss":test_loss,
        "val_correct":correct,
    }

def train(model, train_dataloader, loss_fn, optimizer, epoch_num, val_dataloader=None, save_path=None):
    if torch.cuda.is_available():
        model.cuda()
        loss_fn.cuda()
    best_acc = 0
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    for epoch in range(epoch_num):
        print(f">>> Epoch {epoch+1}/{epoch_num} >>>")
        
        train_dict = train_loop(train_dataloader, model, loss_fn, optimizer)
        epoch_str = f"train_loss:{train_dict['train_Avg_loss']:>7f}\ttrain_acc:{train_dict['train_correct']:>7f}"
        if val_dataloader is not None:
            val_dict = val_loop(val_dataloader, model, loss_fn)
            epoch_str += f"\tval_loss:{val_dict['val_Avg_loss']:>7f}\tval_acc:{val_dict['val_correct']:>7f}"
        print(epoch_str)
        if val_dict['val_correct'] > best_acc:
            best_acc = val_dict['val_correct']
            if save_path is not None:
                torch.save(model.state_dict(), os.path.join(save_path, f"model-{epoch}.pth"))
    return model

def test(model, test_dataloader, loss_fn=None):
    if torch.cuda.is_available():
        model.cuda()
        loss_fn.cuda()
    print(">>> test >>>")
    test_dict = val_loop(test_dataloader, model, loss_fn=loss_fn)
    test_str = f"\ttest_loss:{test_dict['val_Avg_loss']:>7f}\ttest_acc:{test_dict['val_correct']:>7f}"
    print(test_str)
    return test_dict