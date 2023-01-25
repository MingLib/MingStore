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

from MingTorch import set_log_file, path_manager, MDataloader, train
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


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
    from LFW import LFW

    preproceesor=image_preprocesser((250,250), True, True)
    train_dataset=LFW(transform=preproceesor)
    train_loader, val_loader=MDataloader.prepare_data(train_dataset, 128, shuffle=True)
    print(len(train_loader.sampler), len(val_loader.sampler))