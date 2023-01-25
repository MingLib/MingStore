import os
import time
import pytz
import argparse
from datetime import datetime

import glog as log
import torch
from torch import nn
from torchvision import transforms
from torchvision import models
from facenet_pytorch import InceptionResnetV1, MTCNN

from MingTorch import set_log_file, path_manager
from MingTorch.classify import train
from CelebA import CelebA_Align


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def get_model(arch_name):
    if arch_name == "inceptionresnetv1":
        net = InceptionResnetV1('vggface2', classify=True)
        net.requires_grad_(False)
        net.logits = nn.Linear(512, CelebA_Align.classes_num).requires_grad_(True)
    if arch_name == "resnet50":
        net = models.resnet50(True)
        net.requires_grad_(False)
        net.fc = nn.Linear(512*4, CelebA_Align.classes_num).requires_grad_(True)
    if arch_name == "densenet121":
        net = models.densenet121(True)
        net.requires_grad_(False)
        net.classifier = nn.Linear(64, CelebA_Align.classes_num).requires_grad_(True)
    if arch_name == "vgg":
        net = models.vgg11_bn(True)
        net.requires_grad_(False)
        net.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, CelebA_Align.classes_num),
        ).requires_grad_(True)
    #else:
    #    raise ValueError(f"{arch_name} is not found.")
    return net

def prase_args():
    praser=argparse.ArgumentParser()
    praser.add_argument("--cuda",required=True,type=str)
    praser.add_argument("--lr", type=float, default=0.001)
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
    #mtcnn = MTCNN(
    #    image_size=160, margin=0, min_face_size=20,
    #    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True)
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    
    paths_in_train = path_manager(arch_name, "CelebA-Align")
    set_log_file(paths_in_train.record_path)
    
    train_dataset = CelebA_Align('train', data_transform["train"])
    val_dataset = CelebA_Align('val', data_transform["val"])
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                          args.batch_size, 
                                          shuffle=True, 
                                          num_workers=nw,
                                          pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, 
                                          args.batch_size, 
                                          shuffle=True, 
                                          num_workers=nw,
                                          pin_memory=True)
    
    loss_function = nn.CrossEntropyLoss()
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    
    net = train(net, train_loader, loss_function, optimizer, args.epochs, val_loader,f"Models/checkpoints/{arch_name}")
    torch.save(net.state_dict(), paths_in_train.save_path)
    
    end_ = time.time()
    log.info("training time:{time}s".format(time = int(end_ - start_)))
    

if __name__ == "__main__":
    main()

