import codecs
import json
import os
import os.path as osp
from glob import glob

import cv2
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class Dogs(Dataset):
    def __init__(self, args):
        self.paths = glob(osp.join(args.data_path, 'Validation', '*', '**')) + glob(
            osp.join(args.data_path, 'Training', '*', '**'))
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        if os.path.exists(image_path):

            image = cv2.imread(image_path)
            image = cv2.resize(image)
            image = self._transform(image)
            return image
        else:
            print('wrong path')
