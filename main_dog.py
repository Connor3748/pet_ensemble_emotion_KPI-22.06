import os
import json
import random
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
import imgaug
import torch
import torch.multiprocessing as mp
import numpy as np

EMOTION_DICT = {
    0: "행복/놀람(행복/즐거움)",
    1: "중립(편안/안정)",
    2: "두려움/슬픔(불안/슬픔/공포)",
    3: "화남/싫음(화남/불쾌/공격성)",
}
seed = 1234
random.seed(seed)
imgaug.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from classification import models
from classification.models import segmentation

from classification.tta_trainer import FER2013Trainer
def main(config_path):
    """
    This is the main function to make the training up

    Parameters:
    -----------
    config_path : srt
        path to config file
    """
    # load configs and set random seed
    configs = json.load(open(config_path))
    configs["cwd"] = os.getcwd()
    configs["num_classes"] = len(EMOTION_DICT)
    torch.cuda.set_device(configs['device'])
    # load model and data_loader
    model = get_model(configs)

    train_set, val_set, test_set = get_dataset(configs)

    # init trainer and make a training

    print('train start')
    # from trainers.centerloss_trainer import FER2013Trainer
    trainer = FER2013Trainer(model, train_set, val_set, test_set, configs)

    if configs["distributed"] == 1:
        ngpus = torch.cuda.device_count()
        mp.spawn(trainer.train, nprocs=ngpus, args=())
    else:
        trainer.train()


def get_model(configs):
    """
    This function get raw models from models package

    Parameters:
    ------------
    configs : dict
        configs dictionary
    """
    try:
        return models.__dict__[configs["arch"]]
    except KeyError:
        return segmentation.__dict__[configs["arch"]]


def get_dataset(configs):
    """
    This function get raw dataset
    """
    # from utils.datasets.fer2013dataset import fer2013
    from classification.dogemotion import dognet
    # todo: add transform
    train_set = dognet("train", configs)
    val_set = dognet("val", configs)
    test_set = dognet("test", configs)
    # train_set.__getitem__(500)
    print('set')
    return train_set, val_set, test_set

if __name__ == "__main__":
    main("./configs/dog_config.json")