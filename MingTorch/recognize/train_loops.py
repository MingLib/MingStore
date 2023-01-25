import os,sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .evaluate import calculate_roc
from .utils import hflip_batch


torch.backends.cudnn.enable =True

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

@torch.no_grad()
def val_loop(dataloader, model, tta=False):
    model.eval()
    size = len(dataloader.sampler)
    num_batches = len(dataloader)

    pbar = tqdm(total=size)
    features_1, features_2, issame_list = [], [], []
    for batch, ((x1, x2), y) in enumerate(dataloader):
        if torch.cuda.is_available():
            x1 = x1.cuda()
            x2 = x2.cuda()
        feature_map_1 = model(x1)
        feature_map_2 = model(x2)
        if tta:
            fliped1 = hflip_batch(x1)
            fliped2 = hflip_batch(x2)
            emb_batch1 = model(fliped1)
            emb_batch2 = model(fliped2)
            feature_map_1 = l2_norm(feature_map_1 + emb_batch1)
            feature_map_2 = l2_norm(feature_map_2 + emb_batch2)
        features_1.append(feature_map_1)
        features_2.append(feature_map_2)
        issame_list.append(y[0])
            
        pbar.set_description(f"[val] batch:[{batch+1:>3d}/{num_batches:>3d}]")
        pbar.update(len(x1))
    features_1 = torch.cat(features_1).detach().cpu().numpy()
    features_2 = torch.cat(features_2).detach().cpu().numpy()
    issame_list = torch.cat(issame_list).detach().cpu().numpy()
    #
    thresholds = np.arange(0, 4, 0.01)
    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, features_1, features_2,
                                       issame_list, nrof_folds=5, pca=0)
    return {
        "tpr": tpr,
        "fpr": fpr,
        "acc": accuracy.mean(),
        "best_t":best_thresholds.mean(),
    }

def train_loop(dataloader, model, loss_head, optimizer, ce_loss):
    model.train()
    size = len(dataloader.sampler)
    num_batches = len(dataloader)
    train_loss = 0
    
    pbar = tqdm(total=size)
    for batch, (X, y) in enumerate(dataloader):
        if torch.cuda.is_available():
            X = X.cuda()
            y = y.cuda()
        # Compute prediction and loss
        embeddings = model(X)
        logits = loss_head(embeddings, y)
        loss = ce_loss(logits, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        pbar.set_description(f"[train] loss: {loss:>7f}  [{batch+1:>3d}/{num_batches:>3d}]")
        pbar.update(len(X))
    
    train_loss /= num_batches
    return {
        "train_Avg_loss":train_loss,
    }

def train(model, train_dataloader, val_dataloader, loss_head, optimizer, epoch_num, 
          save_path=None, ce_loss=nn.CrossEntropyLoss(), scheduler=None):
    if torch.cuda.is_available():
        model.cuda()
        loss_head.cuda()
    best_acc = 0
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
    for epoch in range(epoch_num):
        print(f">>> Epoch {epoch+1}/{epoch_num} >>>")
        train_dict = train_loop(train_dataloader, model, loss_head, optimizer, ce_loss)
        epoch_str = f"train_loss:{train_dict['train_Avg_loss']:>7f}\t"
        val_dict = val_loop(val_dataloader, model)
        epoch_str += f"val_acc:{val_dict['acc']:>7f}\tbest_threshold:{val_dict['best_t']:>7f}"        
        if scheduler is not None:
            scheduler.step()
        print(epoch_str)
        if val_dict['acc'] > best_acc:
            best_acc = val_dict['acc']
            if save_path is not None:
                checkpoint_path = os.path.join(save_path, f"checkpoint-{epoch}")
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(checkpoint_path, f"model-{epoch}.pth"))
                torch.save(loss_head.state_dict(), os.path.join(checkpoint_path, f"head-{epoch}.pth"))
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, f"optimizer-{epoch}.pth"))
    return model, loss_head, optimizer