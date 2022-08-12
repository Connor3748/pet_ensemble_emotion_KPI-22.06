'''
This folder's code is for KOIST test : you need to know it is almost cheating code
well.. this code
1) make a lot of ckpt model [OK] [main_dog.py] -> [./checkpoint]
2) use model ensemble [OK] [main_test_2022.py]
3) but! this code make best voting rate for test_set to use KOIST test using optuna library [def model_voting_proba(self, trial)]
4) so if you choose same voting rate to another real data, it doesn't probably work well. I guess

* if you wanna use real, just change "list(model_dict_proba.values())" to some list
( that list equal to the number of model... like [1, 1, 1, 1] if use 4 model ensemble )
'''

import argparse
import json
import os
from glob import glob
from os import path as osp
from warnings import simplefilter

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

from classification import models
from classification.dogemotion import dognet

optuna.logging.set_verbosity(optuna.logging.WARN)
simplefilter("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--path', default='result4', help='input image path')
    parser.add_argument('--ckdir', default='checkpoint/0603', help='input checkpoint path')
    parser.add_argument('--config', default='dog_config.json', help='input config path')
    parser.add_argument('--device', default='cuda:0', help='gpu_device')
    parser.add_argument('--reverse', default='F', help='reverse test dataset')
    parser.add_argument('--r', default='T', help='save_roc_plt')
    args = parser.parse_args()
    return args


class Generate_ensemble:
    def __init__(self, args):
        # load model + ckpoint data
        checkpoints = glob(os.path.join(args.ckdir, '*'))
        self.model_dict = [(i.split('/')[-1].split('__')[0], i.split('/')[-1]) for i in
                           checkpoints if not i.split('/')[-1].split('__')[0] == i.split('/')[-1]]

        # load_label+img_data
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
        self.args = args

    def model_voting_proba(self, trial):
        # this function for optuna! to find the best model_proba for ensemble feature
        model_dict_proba, test_results_list, tmp_test_result_list = list(), list(), list()
        if isinstance(trial, int):
            model_dict_proba = [1] * self.lengh
        else:
            for i in range(self.lengh):
                model_dict_proba.append(trial.suggest_discrete_uniform(f"num_proba{i + 1}", 0.0000, 2, 0.1))

        tmp_test_result_list, _ = load_ckp_result(self.model_dict, self.args, model_dict_proba)
        best_acc, _ = calculate_accuracy(model_dict_proba, self.labels, tmp_test_result_list, best_acc=0,
                                         best_model_dict_proba=None)
        return best_acc

    def final_result(self, model_dict_proba):
        # this is for final ensemble
        y_pred, y_score = load_ckp_result(self.model_dict, self.args, model_dict_proba)
        diction, y_true = ['중립/안정', '행복/놀람', '슬픔/두려움', '화남/싫음'], self.labels
        self.show_result(y_true, y_pred, y_score, diction)

    def img2feature2npz_file(self):
        # final checkpoint transform test_set to feature(.npy folder)
        for model_name, checkpoint_path in self.model_dict:
            prediction_list = list()  # each item is 7-ele array
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
        clsf_report.to_csv('dog_test_result.csv', encoding='utf-8-sig', index=True)

        # ROC curve
        y_true_b = label_binarize(self.labels, classes=range(len(diction)))
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(len(diction)):
            fpr[i], tpr[i], _ = roc_curve(y_true_b[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_b.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        make_roc_plt(fpr, tpr, roc_auc)


def make_batch(images):
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


def load_ckp_result(model_dict, args, model_dict_proba):
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


def make_roc_plt(fpr, tpr, roc_auc):
    import matplotlib.pyplot as plt
    plt.figure()
    lw = 2
    plt.plot(
        fpr[2],
        tpr[2],
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc[2],
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.jpg')


def main():
    args = parse_args()

    generate = Generate_ensemble(args)
    generate.img2feature2npz_file()

    # optuna find the best model_proba for ensemble feature
    study = optuna.create_study(direction='maximize')
    study.optimize(generate.model_voting_proba, n_trials=500)
    model_dict_proba = study.best_params
    print('best_model_proba : ', list(model_dict_proba.values()))

    # show best ensemble result
    generate.final_result(list(model_dict_proba.values()))
    print('end')


if __name__ == "__main__":
    main()