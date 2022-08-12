import codecs
import json
import os
import random
from glob import glob

import cv2
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from classification.augment import seg

EMOTION_DICT = {
    0: "행복/놀람(행복/즐거움)",
    1: "중립(편안/안정)",
    2: "두려움/슬픔(불안/슬픔/공포)",
    3: "화남/싫음(화남/불쾌/공격성)",
}


class DogNet(Dataset):
    def __init__(self, stage, configs, tta=False, tta_size=48):
        self._stage = stage
        self._configs = configs
        self._tta = tta
        self._tta_size = tta_size

        self._image_size = (configs["image_size"], configs["image_size"])
        self.path = 'cropdata/final6'
        '''
        plus_trainset = val[:len(val) // 3 * 2]
        validation_set = val[len(val) // 3 * 2:][0::2]
        '''
        val = glob(os.path.join(self.path, "val/*.jpg"))
        plus_trainset = val[:len(val) // 3 * 2]
        validation_set = val[len(val) // 3 * 2:]
        test_set = glob(os.path.join(self.path, 'testset/*.jpg'))
        self.diction = ['행복/즐거움', '편안/안정', '불안/슬픔', '화남/불쾌', '공포', '공격성']
        if stage == "train":
            self.img_path = glob(os.path.join(self.path, "tra/*.jpg"))
            self.img_path = self.img_path + plus_trainset
            self.label_path = '../dataset/dog/Training'

        elif stage == "val":
            self.img_path = validation_set
            self.label_path = '../dataset/dog/Validation'

        elif stage == "test":
            self.img_path = test_set
            self.label_path = '../dataset/dog/Validation'
        else:
            raise Exception("just train or val or test")

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

    def is_tta(self):
        return self._tta == True

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        image_path = self.img_path[idx]

        label, labelpath = self.labelfind(image_path)
        if os.path.exists(labelpath) and os.path.exists(image_path):

            image = cv2.imread(image_path)
            image = cv2.resize(image, self._image_size)

            if self._stage == "train":
                image = seg(image=image)
                image = self._transform(image)
            elif self._stage == "val":
                image = self._transform(image)
            elif self._stage == "test" and self._tta == True:
                #images = [image for i in range(self._tta_size)]
                image = self._transform(image)

            return image, label
        else:
            print('no_label')

    def labelfind(self, image_path):
        labelpath = os.path.join(self.label_path, image_path.split('/')[-1].split('__')[0] + '.json')
        if image_path.split('/')[-2] == 'val':
            labelpath = labelpath.replace('Training', 'Validation')
        if os.path.exists(labelpath) and os.path.exists(image_path):
            f = codecs.open(labelpath, 'r')
            data = json.load(f)
            label = self.diction.index(data['metadata']['inspect']['emotion'])
            if label > 3:
                label = 2 if label == 4 else 3
            return label, labelpath


def list_chunk(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def dognet(stage, configs=None, tta=False, tta_size=48):
    return DogNet(stage, configs, tta, tta_size)

#
# class DogNettest(Dataset):
#     def __init__(self, stage, configs, tta=False, tta_size=48):
#         self._stage = stage
#         self._configs = configs
#         self._tta = tta
#         self._tta_size = tta_size
#
#         self._image_size = (configs["image_size"], configs["image_size"])
#         basepath = '../../../'
#         self.img_path = glob(os.path.join(self.path, "val/*.jpg")) + glob(os.path.join(self.path, "tra/*.jpg"))
#         self.label_path = basepath + 'dataset/dog/Validation'
#
#         self.diction = ['행복/즐거움', '편안/안정', '불안/슬픔', '화남/불쾌', '공포', '공격성']
#         self._transform = transforms.Compose(
#             [
#                 transforms.ToPILImage(),
#                 transforms.ToTensor(),
#             ]
#         )
#
#     def is_tta(self):
#         return self._tta == True
#
#     def __len__(self):
#         return len(self.img_path)
#
#     def __getitem__(self, idx):
#         image_path = self.img_path[idx]
#         # labelpath = os.path.join(self.path,image_path.split('/')[-3],image_path.split('/')[-2] + '.json')
#         label, labelpath = self.labelfind(image_path)
#         if os.path.exists(labelpath) and os.path.exists(image_path):
#             image = cv2.imread(image_path)
#             image = cv2.resize(image, self._image_size)
#
#             if self._stage == "test" and self._tta == True:
#                 images = [seg(image=image) for i in range(self._tta_size)]
#                 image = list(map(self._transform, images))
#
#             return image, label
#
#     def labelfind(self, image_path):
#         labelpath = os.path.join(self.label_path, image_path.split('/')[-1].split('__')[0] + '.json')
#         if image_path.split('/')[-2] == 'val':
#             labelpath = labelpath.replace('Training', 'Validation')
#         if image_path.split('/')[-2] == 'tra':
#             labelpath = labelpath.replace('Validation', 'Training')
#         if os.path.exists(labelpath) and os.path.exists(image_path):
#             f = codecs.open(labelpath, 'r')
#             data = json.load(f)
#             label = self.diction.index(data['metadata']['inspect']['emotion'])
#             if label > 3:
#                 label = 2 if label == 4 else 3
#             return label, labelpath
#
#
# def dognettest(stage, configs=None, tta=False, tta_size=48):
#     return DogNettest(stage, configs, tta, tta_size)
