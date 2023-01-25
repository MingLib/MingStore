import os
import time
import pytz
import argparse
from datetime import datetime
from pathlib import Path

import glog as log
import torch
from torch import nn
from torch import optim
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms
from torchvision import models
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch.optim.lr_scheduler as lr_scheduler

from MingTorch import set_log_file, path_manager
from MingTorch.recognize import train
from MingTorch.recognize.utils import separate_bn_paras
from CelebA import CelebA_Align
from Vgg_Face2 import VggFace2, WarpAffine
from LFW import LFW
from InsightFace_Pytorch.model import Arcface, MobileFaceNet
from MingTorch.recognize import ArcMarginProduct


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def get_model(arch_name):
    if arch_name == "inceptionresnetv1":
        net = InceptionResnetV1('vggface2', classify=False)
        net.requires_grad_(True)
    elif arch_name == "mobilefacenet":
        net = MobileFaceNet(512)
        net.requires_grad_(True)
    return net

def prase_args():
    praser=argparse.ArgumentParser()
    praser.add_argument("--cuda",required=True,type=str)
    praser.add_argument("--lr", type=float, default=1e-3)
    praser.add_argument("--arch", required=True, type=str)
    praser.add_argument("-e", "--epochs", type=int, default=100)
    praser.add_argument("-b", "--batch_size", default=64, type=int)
    args=praser.parse_args()
    
    date = datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    log.info("date:{date}".format(date = date))
    log.info("training {net}".format(net = args.arch))
    log.info("epochs:{num}".format(num = args.epochs))
    log.info("batch_size:{size}".format(size=args.batch_size))
    log.info("optimizer ADAM with learning rate {lr}".format(lr = args.lr))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    return args

def main():
    args = prase_args()
    arch_name = args.arch
    start_ = time.time()
    
    net = get_model(args.arch)
    mtcnn = MTCNN(
        image_size=112, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True)
    data_transform = {
        "train": transforms.Compose([#transforms.RandomResizedCrop(112),
                                     #transforms.Resize(112),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([#transforms.Resize(112),
                                   transforms.CenterCrop(112),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    
    paths_in_train = path_manager(arch_name, "vggface2-Align")
    model_path = Path(paths_in_train.save_path[:-4])
    set_log_file(paths_in_train.record_path)
    
    #train_dataset = CelebA_Align('train', data_transform["train"])
    train_dataset = VggFace2('train', transform=data_transform["train"], img_aligen=WarpAffine.similarity, crop_size=(112,112))
    val_dataset = LFW(data_transform["val"])
    #nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    nw = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                          args.batch_size, 
                                          shuffle=False, 
                                          num_workers=nw,
                                          pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                          args.batch_size, 
                                          shuffle=True, 
                                          num_workers=nw,
                                          pin_memory=True)
    
    ce_loss_function = nn.CrossEntropyLoss()
    loss_head = Arcface(classnum=CelebA_Align.classes_num)
    # construct an optimizer
    paras_only_bn, paras_wo_bn = separate_bn_paras(net)
    optimizer = optim.SGD([
                        {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                        {'params': [paras_wo_bn[-1]] + [loss_head.kernel], 'weight_decay': 4e-4},
                        {'params': paras_only_bn}
                    ], lr = args.lr, momentum = 0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [12,15,18], 0.1)
    
    net, loss_head, optimizer = train(net, train_loader, val_loader, loss_head, optimizer, args.epochs, model_path, ce_loss_function, scheduler)
    torch.save(net.state_dict(), model_path/"model.pth")
    torch.save(loss_head.state_dict(), model_path/"head.pth")
    torch.save(optimizer.state_dict(), model_path/"optimizer.pth")
        
    end_ = time.time()
    log.info("training time:{time}s".format(time = int(end_ - start_)))
    

if __name__ == "__main__":
    main()

