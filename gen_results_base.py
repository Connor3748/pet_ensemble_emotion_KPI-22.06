import os
import random
import json
import imgaug
import torch
import numpy as np

seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from tqdm import tqdm
import models
import torch.nn.functional as F
from utils.datasets.fer2013dataset import fer2013
from utils.datasets.dogemotion import dognet, dognettest
from utils.generals import make_batch

model_dict = [
    ("resmasking", "resmasking_test_remasking_2022Feb11_05.24"),
    ("resmasking", "resmasking_test_resmasking_0225_2022Feb25_09.01"),
    ("resmasking", "resmasking_test_resmasking_0304_2022Mar04_06.12"),
    ("resmasking", "resmasking_test_resmasking_0305_2022Mar07_00.35"),
    ("inception_v3", "inception_v3_test_inception_v30225_2022Feb25_08.58"),
    ("inception_v3", "inception_v3_test_inception_v3_0304_2022Mar04_06.22"),
    ("inception_v3", "inception_v3_test_inception_v3_0305_2022Mar07_00.34"),
    ("inception_v3", "inception_v3_test_inception_v3_0307_2022Mar07_11.49"),
    ("resnet152", "resnet152_test_resnet152_0225_2022Feb25_09.00"),
    ("resnet152", "resnet152_test_resnet152_0305_2022Mar07_00.33"),
    ("resnet152", "resnet152_test_resnet152_0305_2022Mar07_00.33"),
    ("resnet152", "resnet152_test_resnet152_0307_2022Mar07_11.52"),
    ("resnet152", "resnet152_test_resnet152pretrained_0221_2022Feb21_05.39"),
    ("resnet152", "resnet152_test_resnet152pretrained_2022Feb11_05.44"),
    ("densenet161", "densenet161_test_densenet161_0225_2022Feb25_09.09"),
    ("densenet161", "densenet161_test_densenet161_0307_2022Mar07_11.47"),
    ("densenet161", "densenet161_test_densenet161mixup_0224_2022Feb24_06.51"),
    ("cbam_resnet50", "cbam_resnet50_test_cbam_2022Feb11_02.00"),
    ("cbam_resnet50", "cbam_resnet50_test_cbam_resnet50_0225_2022Feb25_09.08"),
    ("cbam_resnet50", "cbam_resnet50_test_cbam_resnet50_0305_2022Mar07_00.36"),
    ("cbam_resnet50", "cbam_resnet50_test_cbam_resnet50mixup_0225_2022Feb25_01.48"),
]

def main():
    with open("../configs/dog_config.json") as f:
        configs = json.load(f)

    test_set = dognettest("test", configs, tta=True, tta_size=8)
    for model_name, checkpoint_path in model_dict:
        prediction_list = []  # each item is 7-ele array

        print("Processing", checkpoint_path)
        if os.path.exists("./saved/results/{}.npy".format(checkpoint_path)):
            continue

        model = getattr(models, model_name)
        model = model(in_channels=3, num_classes=6)

        state = torch.load(os.path.join("../checkpoint", checkpoint_path))
        model.load_state_dict(state["net"])

        model.cuda()
        model.eval()

        with torch.no_grad():
            for idx in tqdm(range(len(test_set)), total=len(test_set), leave=False):
                images, targets = test_set[idx]
                images = make_batch(images)
                images = images.cuda(non_blocking=True)

                outputs = model(images).cpu()
                outputs = F.softmax(outputs, 1)
                outputs = torch.sum(outputs, 0)  # outputs.shape [tta_size, 7]

                outputs = [round(o, 4) for o in outputs.numpy()]
                prediction_list.append(outputs)

        np.save("../saved/results/{}.npy".format(checkpoint_path), prediction_list)


if __name__ == "__main__":
    main()
