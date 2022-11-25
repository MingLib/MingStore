import os,sys
sys.path.append(os.getcwd())

import torch
import numpy as np
from tqdm import tqdm

torch.backends.cudnn.enable =True

def train_loop(dataloader, model, loss_fn, optimizer):
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
        
        pbar.set_description(f"loss: {loss.item():>7f}  [{batch+1:>3d}/{num_batches:>3d}]")
        pbar.update(len(X))
    
    train_loss /= num_batches
    correct /= size
    return {
        "train_Avg_loss":train_loss,
        "train_correct":correct,
    }

def val_loop(dataloader, model, loss_fn):
    size = len(dataloader.sampler)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    return {
        "val_Avg_loss":test_loss,
        "val_correct":correct,
    }

def train(model,train_dataloader,loss_fn,optimizer,epoch_num, save_path, val_dataloader=None):
    if torch.cuda.is_available():
        model.cuda()
        loss_fn.cuda()
    net = model.train()
    for epoch in range(epoch_num):
        print(f">>> Epoch {epoch+1}/{epoch_num} >>>")
        
        train_dict = train_loop(train_dataloader, net, loss_fn, optimizer)
        epoch_str = f"train_loss:{train_dict['train_Avg_loss']:>7f}\ttrain_acc:{train_dict['train_correct']:>7f}"
        if val_dataloader is not None:
            val_dict = val_loop(val_dataloader, net, loss_fn)
            epoch_str += f"\tval_loss:{val_dict['val_Avg_loss']:>7f}\tval_acc:{val_dict['val_correct']:>7f}"
        print(epoch_str)
        if epoch %10 == 0:
            torch.save(model.state_dict(), save_path)
    return net

def recognize_evaluate_loop(dataloader, model, dis_fn):
    size = len(dataloader.sampler)
    num_batches = len(dataloader)
    test_score = []

    pbar = tqdm(total=size)
    with torch.no_grad():
        for batch, ((x1, x2), _) in enumerate(dataloader):
            if torch.cuda.is_available():
                x1 = x1.cuda()
                x2 = x2.cuda()
            feature_map_1 = model(x1)
            feature_map_2 = model(x2)
            test_score.append(dis_fn(feature_map_1, feature_map_2).detach().cpu().numpy())
            
            pbar.set_description(f"batch:[{batch+1:>3d}/{num_batches:>3d}]")
            pbar.update(len(x1))

    test_score = np.hstack(test_score)
    return {
        "test_score":test_score,
    }
    
def recognize_evaluate(model, test_dataloader, metirc=None, dis_fn="l2"):
    if dis_fn == "l2":
        dis_fn = lambda x1,x2:torch.norm(x1-x2, dim=1, p=2)
    elif dis_fn == "cos":
        dis_fn = lambda x1,x2:torch.cosine_similarity(x1, x2, dim=1)
        
    if torch.cuda.is_available():
        model.cuda()
    net = model.eval()
        
    print(f">>> predict period >>>")
    evaluate_dict = recognize_evaluate_loop(test_dataloader, net, dis_fn)
    scores = evaluate_dict['test_score']
    if metirc is not None:
        print(f">>> verification period >>>")
        return scores, metirc(scores, test_dataloader)
    else:
        return scores