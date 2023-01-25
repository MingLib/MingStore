import os

import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN

from LFW import LFW
from MingTorch.recognize import lfw_cross_validation ,validation
from MingTorch.recognize.train_loops import val_loop
from backbones import get_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
        
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    
    #net = InceptionResnetV1('casia-webface', classify=False)
    net = get_model('r100')
    #print(torch.load("model.pt"))
    net.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
    
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        )
    transformers = transforms.Compose(
        [transforms.Lambda(lambda img:mtcnn(img)),]
    )
    
    transformers = transforms.Compose([#transforms.Resize(112),
                                   transforms.CenterCrop(112),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                                   #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    

    dataset = LFW(transform=transformers)
    lfw_dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=60)
    
    _, b = validation(net, lfw_dataloader, lfw_cross_validation, dis_fn="cos")
    #val_dict = val_loop(lfw_dataloader, net)
    #print(val_dict['best_t'], val_dict['acc'])
    print(b["accuracy_mean"], b["accuracy"])
    