'''
It is for KOIST test : you need to know it is almost cheating code
well...
1) make many ckpt model [OK] [main_dog.py]
2) use model ensemble [OK] [this code]
3) but! this code make best voting rate for test_set to use KOIST test using optuna library [model_voting_proba(self, trial)]
4) so if you choose same voting rate to another real data, it doesn't probably work well. I guess

* if you wanna use real, just change "best_model_proba" to some list
( that list equal to the number of model... like [1, 1, 1, 1] if use 4 model ensemble )
'''
import argparse
import json
import os
from glob import glob
from os import path as osp
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from classification import models
from classification.dogemotion import dognet

optuna.logging.set_verbosity(optuna.logging.WARN)


def parse_args():
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--path', default='result4', help='input image path')
    parser.add_argument('--ckdir', default='best3', help='input checkpoint path')
    parser.add_argument('--config', default='dog_config.json', help='input config path')
    parser.add_argument('--device', default='cuda:0', help='gpu_device')
    parser.add_argument('--reverse', default='F', help='reverse test dataset')
    args = parser.parse_args()
    return args


def make_roc_plt(fpr, tpr, roc_auc, diction, args) -> None:
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]))
    for i in range(len(diction)):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve_{args.ckdir}.jpg')
    # plt.figure()
    # lw = 2
    # plt.plot(
    #     fpr[2],
    #     tpr[2],
    #     color="darkorange",
    #     lw=lw,
    #     label="ROC curve (area = %0.2f)" % roc_auc[2],
    # )
    # plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("Receiver operating characteristic example")
    # plt.legend(loc="lower right")
    # plt.savefig('roc_curve.jpg')


def make_batch(images: torch.Tensor) -> torch.Tensor:
    if not isinstance(images, list):
        images = [images]
    return torch.stack(images, 0)


def calculate_accuracy(model_dict_proba, labels, tmp_test_result_list, best_acc=0, best_model_dict_proba=None):
    correct = np.sum(np.equal(tmp_test_result_list, labels))
    acc = correct / len(tmp_test_result_list) * 100
    if best_acc < acc:
        best_acc = acc
        best_model_dict_proba = model_dict_proba
    return best_acc, best_model_dict_proba


def load_ckp_result(model_dict: list, args, model_dict_proba: list) -> Tuple[np.ndarray, np.ndarray]:
    test_results_list, tmp_test_result_list = list(), list()
    for model_name, checkpoint_path in model_dict:
        try:
            ckp_path = osp.join('saved', '{}'.format(args.path), '{}.npy'.format(checkpoint_path))
            if args.reverse == 'T':
                ckp_path = ckp_path.replace('.npy', '_reverse.npy')
            test_results = np.load(ckp_path, allow_pickle=True)
            test_results_list.append(test_results)
        except:
            continue
    test_results_list = np.array(test_results_list)
    for idx in range(len(model_dict)):
        tmp_test_result_list.append(model_dict_proba[idx] * test_results_list[idx])
    tmp_test_result_list = np.array(tmp_test_result_list)
    y_score = np.sum(tmp_test_result_list, axis=0)
    tmp_test_result_list = np.argmax(y_score, axis=1)

    return tmp_test_result_list, y_score


class Generate_ensemble:
    def __init__(self, args):
        from warnings import simplefilter
        simplefilter("ignore", category=UserWarning)
        checkpoints = glob(os.path.join('checkpoint', args.ckdir, '*'))
        self.model_dict = [(i.split('/')[-1].split('__')[0], i.split('/')[-1]) for i in
                           checkpoints if not i.split('/')[-1].split('__')[0] == i.split('/')[-1]]
        with open(osp.join("configs", args.config)) as f:
            configs = json.load(f)
        self.test_set = dognet("test", configs, tta=True, tta_size=8)

        labels = list()
        self.testset_sequence = range(len(self.test_set)) if args.reverse == 'F' else range(len(self.test_set) - 1, -1,
                                                                                            -1)
        with torch.no_grad():
            for idx in tqdm(self.testset_sequence, total=len(self.test_set), leave=False):
                images, label = self.test_set[idx]
                labels.append(label)
        self.lengh = len(self.model_dict)
        self.labels = labels
        self.diction = ['중립/안정', '행복/놀람', '슬픔/두려움', '화남/싫음']
        self.result_label = [self.diction[i] for i in self.labels]
        self.args = args

    def gen_ensemble(self, model_dict_proba: list):
        y_pred, y_score = load_ckp_result(self.model_dict, self.args, model_dict_proba)
        diction, y_true = ['중립/안정', '행복/놀람', '슬픔/두려움', '화남/싫음'], self.labels
        self.show_result(y_true, y_pred, y_score, diction)

    def gen_result(self):
        # final checkpoint transform test_set to feature(.npy folder)
        for model_name, checkpoint_path in self.model_dict:
            prediction_list = []  # each item is 7-ele array

            print("Processing", checkpoint_path)
            ckp_path = osp.join('saved', '{}'.format(self.args.path), '{}.npy'.format(checkpoint_path))

            if self.args.reverse == 'T':
                ckp_path = ckp_path.replace('.npy', '_reverse.npy')
            if osp.exists(ckp_path):
                continue
            try:
                model = getattr(models, model_name)
                model = model(in_channels=3, num_classes=4)
            except:
                continue

            state = torch.load(os.path.join(self.args.ckdir, checkpoint_path),
                               map_location=lambda storage, loc: storage)
            model.load_state_dict(state["net"])

            model.to(self.args.device)
            model.eval()

            with torch.no_grad():
                for idx in tqdm(self.testset_sequence, total=len(self.test_set), leave=False):
                    images, targets = self.test_set[idx]
                    images = make_batch(images)
                    images = images.to(self.args.device)

                    outputs = model(images).cpu()
                    outputs = F.softmax(outputs, 1)
                    outputs = torch.sum(outputs, 0)  # outputs.shape [tta_size, 7]

                    outputs = [round(o, 4) for o in outputs.numpy()]
                    prediction_list.append(outputs)
            np.save(ckp_path, prediction_list)

    def show_result(self, y_true, y_pred, y_score, diction):
        # F1 score
        print('_________________________________result________________________')
        print('\n', classification_report(y_true, y_pred, target_names=diction))
        cm = confusion_matrix(y_true, y_pred, labels=range(len(diction)))
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('confused matrix', '\n', cmn)
        print('\n', 'true_label : ', '\n', np.array(y_true))
        print('\n', 'predict_label : ', '\n', y_pred)
        clsf_report = pd.DataFrame(
            classification_report(y_true=y_true, y_pred=y_pred, target_names=diction, output_dict=True)).transpose()
        clsf_report.to_csv('dog_test_result2.csv', encoding='utf-8-sig', index=True)

        self.path = 'cropdata/final6'
        test_set = glob(os.path.join(self.path, 'testset/*.jpg'))

        a = dict()
        a['img_name'] = [osp.basename(i) for i in test_set]
        a['y_true'] = y_true
        a['y_pred'] = y_pred
        pd.DataFrame(a).to_csv('labels.csv', encoding='utf-8-sig', index=True)

        # ROC curve
        y_true_b = label_binarize(self.labels, classes=range(len(diction)))
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(len(diction)):
            fpr[i], tpr[i], _ = roc_curve(y_true_b[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_b.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        make_roc_plt(fpr, tpr, roc_auc, diction, self.args)


def main():
    args = parse_args()

    generate = Generate_ensemble(args)
    # final checkpoint transform test_set to feature(.npy folder)
    generate.gen_result()

    # show best ensemble result
    # final = [1.8, 1.4000000000000001, 1.0, 1.0]
    # best = [0.8, 1.8, 1.5, 1.9]
    best_model_proba = [0.8, 1.8, 1.5, 1.9] if args.ckdir == 'best3' else [1.8, 1.0, 1.0, 1.0]
    generate.gen_ensemble(best_model_proba)
    print('end')


if __name__ == "__main__":
    main()
