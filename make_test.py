
import codecs
import json
import os
from glob import glob

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.transforms import transforms
from tqdm import tqdm

from classification import models
from classification.dogemotion import dognet


def face_sparse(lencheck, count, maketestset):
    if lencheck:
        pass  # if not label == 3 else 10
    elif count:
        del maketestset[-1]
    return count


def make_testset():
    testlen = 100
    ppath = os.path.join('cropdata', 'cat1', "val")
    img_path = glob(os.path.join(ppath, "*.jpg"))
    img_paths = img_path[len(img_path) // 4 * 3:][0::2]
    label_path = '../dataset/dog/Validation' if not 'cat' in ppath else '../dataset/cat/Validation'
    diction = ['행복/즐거움', '편안/안정', '불안/슬픔', '화남/불쾌', '공포', '공격성']
    _transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )
    maketestset, count, filecheck = [10], 1, [10, 9]
    for image_path in img_paths:
        if 'cat' in ppath:
            label_file = image_path.split('/')[-1].split('__')[0]
            label_paths = os.path.join(label_path, label_file.split('-')[1].upper(), label_file + '.json')
        else:
            label_paths = os.path.join(label_path, image_path.split('/')[-1].split('__')[0] + '.json')
        file = image_path.split('/')[-1].split('__')[0].replace('.mp4', '')

        if os.path.exists(label_path) and os.path.exists(image_path):
            f = codecs.open(label_paths, 'r')
            data = json.load(f)
            label = diction.index(data['metadata']['inspect']['emotion'])
            if label > 3:
                label = 2 if label == 4 else 3
            if 'cat' in ppath:
                if label > 1:
                    label = 2
            maketestset.append(label)
            lencheck = (maketestset.count(label) > testlen)

            if maketestset.count(label) > 100 or file in filecheck:
                count = face_sparse(lencheck, count, maketestset)
                continue
            filecheck.append(file)
            image = cv2.imread(image_path)
            # image = cv2.resize(image, (256, 256))
            cv2.imwrite(image_path.replace('val', 'testset'), image)
    for i in range(4):
        print(maketestset.count(i))

    print('end')


def main():
    # make testset
    make_testset()
    # save testset to .npz using checkpoint to make ensemble
    print('end')


if __name__ == "__main__":
    main()
