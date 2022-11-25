import os.path as osp

from torch.utils import data
from torchvision.datasets.folder import default_loader

class LFW(data.Dataset):
    def __init__(self, transform=None) -> None:
        super().__init__()
        root = osp.split(__file__)[0]
        paire_file_path = osp.join(root,"Data/pairs.txt")
        with open(paire_file_path, 'r') as pairs_file:
            pairs_file.readline()
            lines = pairs_file.read().strip('\n').split('\n')
            
        samples, class_names, target = [], [], []
        for line in lines:
            annos = line.strip().split('\t')
            if len(annos) == 3:
                class_names.append([annos[0]])
                imgpth1 = annos[0] + '/' + annos[0] + '_' + str(annos[1]).zfill(4) + '.jpg'
                imgpth2 = annos[0] + '/' + annos[0] + '_' + str(annos[2]).zfill(4) + '.jpg'
                samples.append([imgpth1, imgpth2])
                target.append([1])
            elif len(annos) == 4:
                class_names.append([annos[0], annos[2]])
                imgpth1 = annos[0] + '/' + annos[0] + '_' + str(annos[1]).zfill(4) + '.jpg'
                imgpth2 = annos[2] + '/' + annos[2] + '_' + str(annos[3]).zfill(4) + '.jpg'
                samples.append([imgpth1, imgpth2])
                target.append([0])
        
        self.root = root
        self._target = target
        self.samples = samples
        self.class_names = class_names
        self.transform = transform
    
    def __getitem__(self, index):
        filename1, filename2 = self.samples[index]
        label = self._target[index]
        sample1 = default_loader(osp.join(self.root, "Data/lfw/", filename1))
        sample2 = default_loader(osp.join(self.root, "Data/lfw/", filename2))
        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)
        return [sample1, sample2], label
    
    def __len__(self) -> int:
        return len(self.samples)
    
    @property
    def target(self):
        return self._target