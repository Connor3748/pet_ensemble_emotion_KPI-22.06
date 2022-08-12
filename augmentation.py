import codecs
from glob import glob
import os
import json

import cv2
from imgaug import augmenters as iaa

diction = ['행복/즐거움', '편안/안정', '불안/슬픔', '화남/불쾌', '공포', '공격성']


def labelfind(image_path, label_path):
    labelpath = os.path.join(label_path, image_path.split('/')[-1].split('__')[0] + '.json')
    if image_path.split('/')[-2] == 'val':
        labelpath = labelpath.replace('Training', 'Validation')
    if os.path.exists(labelpath) and os.path.exists(image_path):
        f = codecs.open(labelpath, 'r')
        data = json.load(f)
        label = diction.index(data['metadata']['inspect']['emotion'])
        if label > 3:
            label = 2 if label == 4 else 3
        return label, labelpath


def list_chunk(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]


path = './cropdata/test'
aug = iaa.Sequential(
    [
        iaa.Fliplr(p=0, deterministic=True),
        iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.5, 1.5)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
        iaa.AdditiveLaplaceNoise(scale=(0, 0.05 * 255)),
        iaa.GaussianBlur(sigma=(0., 1.0), deterministic=True),
        iaa.MultiplyHueAndSaturation((0.2, 1.5)),
        iaa.Dropout((0., 0.15), deterministic=True),
    ]
)

img_paths = glob(os.path.join(path, "tra", "*.jpg"))
label_path = '../dataset/dog/Training'

diction = ['행복/즐거움', '편안/안정', '불안/슬픔', '화남/불쾌', '공포', '공격성']

maketestset, count, filecheck = [10], 1, [10, 9]
for image_path in img_paths:
    label_paths = os.path.join(label_path, image_path.split('/')[-1].split('__')[0] + '.json')
    file = image_path.split('/')[-1].split('__')[0].replace('.mp4', '')

    if os.path.exists(label_path) and os.path.exists(image_path):
        f = codecs.open(label_paths, 'r')
        data = json.load(f)
        label = diction.index(data['metadata']['inspect']['emotion'])
        if label > 3:
            label = 2 if label == 4 else 3
        if label > 1:
            maketestset.append(label)
            filecheck.append(file)
            image = cv2.imread(image_path)
            image = aug(image=image)
            cv2.imwrite(image_path.replace('__', '__ann'), image)

img_path = glob(os.path.join(path, "tra/*.jpg"))
img_path = img_path
label_path = '../dataset/dog/Training'
labelcount = []
for i in range(len(img_path)):
    image_path = img_path[i]
    label, _ = labelfind(image_path, label_path)
    labelcount.append(label)
for i in range(4):
    print(labelcount.count(i))
print('end')
