import torch
import numpy as np
from tqdm import tqdm

def validation_loop(dataloader, model, dis_fn):
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
            
            pbar.set_description(f"[val] batch:[{batch+1:>3d}/{num_batches:>3d}]")
            pbar.update(len(x1))

    test_score = np.hstack(test_score)
    return {
        "test_score":test_score,
    }
    
def validation(model, test_dataloader, metirc=None, dis_fn="l2"):
    if dis_fn == "l2":
        dis_fn = lambda x1,x2:torch.norm(x1-x2, dim=1, p=2)
    elif dis_fn == "cos":
        dis_fn = lambda x1,x2:torch.cosine_similarity(x1, x2, dim=1)
        
    if torch.cuda.is_available():
        model.cuda()
    net = model.eval()
        
    print(f">>> predict period >>>")
    evaluate_dict = validation_loop(test_dataloader, net, dis_fn)
    scores = evaluate_dict['test_score']
    if metirc is not None:
        print(f">>> verification period >>>")
        return scores, metirc(scores, test_dataloader)
    else:
        return scores

def lfw_cross_validation(scores, dataloader):
    
    def get_rate(scores_list, y, t):
        pos_score_list = scores_list[y == 1]
        neg_socre_lsit = scores_list[y == 0]
        pso_num = len(pos_score_list)
        neg_num = len(neg_socre_lsit)
        pso_succ_num = np.sum(pos_score_list > t) 
        neg_fail_num = np.sum(neg_socre_lsit > t)
        tpr = pso_succ_num / pso_num
        fpr = neg_fail_num / neg_num
        accu = (pso_succ_num + neg_num - neg_fail_num) / len(scores_list)
        return accu, tpr, fpr
    
    def getThreshold(scores_list, y):
        thresholds = np.linspace(np.min(scores_list), np.max(scores_list), 1000)
        tpr_list, fpr_list = [], []
        for t in thresholds:
            _, tpr, fpr = get_rate(scores_list, y, t)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        tpr_list = np.array(tpr_list)
        fpr_list = np.array(fpr_list)
        best_index = np.argmax(tpr_list- fpr_list)
        best_t = thresholds[best_index]
        return best_t
    
    targets = np.array(dataloader.dataset.target)
    targets = np.squeeze(targets) 
    score_list_index = np.arange(0, len(scores))
    accu_list = []
    thresholds = []
    for test_list_index in np.split(score_list_index, 10):
        train_list_index = np.setdiff1d(score_list_index, test_list_index)
        train_list = scores[train_list_index]
        train_label_list = targets[train_list_index]
        test_list = scores[test_list_index]
        test_label_list = targets[test_list_index]
        t = getThreshold(train_list, train_label_list)
        accu, _, _ = get_rate(test_list, test_label_list, t)
        accu_list.append(accu)
        thresholds.append(t)
        
    return {
        "accuracy_mean": np.mean(accu),
        "accuracy_std": np.std(accu),
        "accuracy": np.array(accu_list),
        "thresholds": np.array(thresholds),
    }