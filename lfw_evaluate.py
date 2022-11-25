import os

import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN

from LFW import LFW
from torch_loop import recognize_evaluate, lfw_cross_validation

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            
        
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    
    #net = InceptionResnetV1('casia-webface', classify=False)
    net = InceptionResnetV1('vggface2', classify=False)
    
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        )
    transformers = transforms.Compose(
        [transforms.Lambda(lambda img:mtcnn(img)),]
    )
    
    dataset = LFW(transform=transformers)
    lfw_dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=60)
    
    _, b = recognize_evaluate(net, lfw_dataloader, lfw_cross_validation, dis_fn="cos")
    print(b['accuracy_mean'])
    