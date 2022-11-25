import os
import time
import pytz
import argparse
from datetime import datetime

import glog as log
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from facenet_pytorch import InceptionResnetV1, MTCNN

from torch_loop import set_log_file, path_manager, prepare_data, train
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

LFW_DATASET={
    "name":"lfw",
    "image_size":(250, 250),
    "num_classes":5749,
    "root":os.path.expanduser(r"./datasets/lfw/")
}

mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        )

def image_preprocesser(input_size,use_flip=False,center_crop =False):
    processors = []
    if input_size is not None:
        processors.append(transforms.Resize(size=input_size))
    if use_flip:
        processors.append(transforms.RandomHorizontalFlip())
    if center_crop:
        processors.append(transforms.CenterCrop(max(input_size)))
    #processors.append(lambda x:mtcnn(x))
    #processors.append(transforms.ToTensor())
    return transforms.Compose(processors)   

if __name__ == "__main__":
    date = datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    start_ = time.time()

    praser=argparse.ArgumentParser()
    praser.add_argument("--cuda",required=True,type=str)
    praser.add_argument("--lr", type=float, default=0.001)
    praser.add_argument("-e", "--epochs", type=int, default=100)
    praser.add_argument("-b", "--batch_size", default=64, type=int)
    args=praser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    
    arch_name='InceptionResnetv1'

    cuda_visible_devices = int(args.cuda)

    paths_in_train = path_manager(arch_name, "LFW")

    set_log_file(paths_in_train.record_path)

    log.info("date:{date}".format(date = date))
    log.info("Using GPU:{id}".format(id = args.cuda))
    log.info("under {frame} frame".format(frame = "torch"))
    log.info("training {net}".format(net = arch_name))
    log.info("epochs:{num}".format(num = args.epochs))
    log.info("batch_size:{size}".format(size=args.batch_size))
    log.info("optimizer ADAM with learning rate {lr}".format(lr = args.lr))

    dataset = LFW_DATASET
    
    net = InceptionResnetV1('casia-webface', classify=True).eval()
    net.logits = nn.Linear(512, dataset['num_classes'])
    #print(net)

    preproceesor=image_preprocesser(dataset["image_size"], True, True)
    train_dataset=ImageFolder(dataset["root"],transform=preproceesor)
    train_loader, val_loader=prepare_data(train_dataset, args.batch_size, shuffle=True)
    
    loss=torch.nn.CrossEntropyLoss()
    Optimizer=torch.optim.Adam(net.parameters(),lr=args.lr)

    net = train(net, train_loader, loss, Optimizer, args.epochs, paths_in_train.save_path, val_loader)
    
    torch.save(net.state_dict(), paths_in_train.save_path)

    end_ = time.time()
    log.info("training time:{time}s".format(time = int(end_ - start_)))