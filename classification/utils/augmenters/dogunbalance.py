import imgaug
from imgaug import augmenters as iaa
import random

imgaug.seed(1)

seg2 = iaa.Sequential(
    [
        # iaa.Fliplr(p=0.5, deterministic=True),
        # iaa.Affine(rotate=(-30, 30), deterministic=True),
        # iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.5, 1.5)),
        # iaa.PerspectiveTransform(scale=(0.01, 0.03)),
        # iaa.PiecewiseAffine(scale=(0.01, 0.02)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
        iaa.AdditiveLaplaceNoise(scale=(0, 0.05*255)),
        # iaa.Dropout((0., 0.15), deterministic=True),
        # iaa.Add((-25, 25), deterministic=True),
        # iaa.CropAndPad(percent=(-0.05, 0.1), pad_cval=(0, 255), deterministic=True)
    ]
)