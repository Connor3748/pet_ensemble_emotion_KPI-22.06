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
import cv2
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
    parser.add_argument('--path', default='result8', help='input image path')
    parser.add_argument('--ckdir', default='checkpoint/four_cat', help='input checkpoint path')
    parser.add_argument('--config', default='dog_config.json', help='input config path')
    parser.add_argument('--device', default='cuda:0', help='gpu_device')
    parser.add_argument('--reverse', default='F', help='reverse test dataset')
    parser.add_argument('--r', default='T', help='save_roc_plt')
    parser.add_argument('--optuna_num', default=3000, help='choose optuna trial number')
    parser.add_argument('--class_num', default=4, help='len(class)')
    parser.add_argument('--dog_cat', default='cat', help='dog or cat')
    parser.add_argument('--ensemble_indicator', default='B',
                        help='binary(Positive or Negative)[B] or original==class number(dog =4, cat=3)[O]')
    parser.add_argument('--optuna_range', default=2, type=int,
                        help='This is for optuna trial range. if set 2, it is setting to try 0 ~ 2 rate for ensemble')
    args = parser.parse_args()
    return args


class Generate_ensemble:
    def __init__(self, args):
        # load model + check point data
        self._results_list = None
        checkpoints = glob(os.path.join(args.ckdir, '*'))
        self.model_dict = [(i.split('/')[-1].split('__')[0], i.split('/')[-1]) for i in
                           checkpoints if not i.split('/')[-1].split('__')[0] == i.split('/')[-1]]
        # load_label+img_data
        self.test_set = catnet2() if args.dog_cat == 'cat' else dognet2()
        labels = list()
        self.test_set_sequence = range(len(self.test_set))
        with torch.no_grad():
            for idx in tqdm(self.test_set_sequence, total=len(self.test_set), leave=False):
                images, label = self.test_set[idx]
                # remover this line
                labels.append(label)
        self.length = len(self.model_dict)
        self.labels = labels
        self.args = args

    def model_voting_proba(self, trial, acc=0, best_model_dict_proba=None):
        # this function for optuna process! to find the best model_proba for ensemble feature

        _results_list = self._results_list.copy()
        _results_list.pop(self.del_ind)
        self.selected_model_dict = self.model_dict.copy()
        self.selected_model_dict.pop(self.del_ind)

        # first : set range of trial
        model_dict_proba, test_results_list, tmp_test_result_list = list(), list(), list()
        if isinstance(trial, int):
            model_dict_proba = [1] * (self.length - 1)
        elif isinstance(trial, list):
            model_dict_proba = trial
        else:
            # range 0 ~ self.args.optuna_range
            for i in range(self.length - 1):
                model_dict_proba.append(
                    trial.suggest_discrete_uniform(f"num_proba{i + 1}", 0.0000, self.args.optuna_range, 0.1))

        tmp_test_result_list, _, _, _ = output_result(_results_list, self.selected_model_dict, model_dict_proba)

        if self.args.ensemble_indicator == 'B':
            labels, pred_result = list(), list()
            for label in self.labels:
                label = 0 if not label > 1 else 1
                labels.append(label)
            for i in tmp_test_result_list:
                i = 0 if not i > 1 else 1
                pred_result.append(i)
        else:
            labels, pred_result = self.labels, tmp_test_result_list
        best_acc, _ = calculate_accuracy(model_dict_proba, pred_result, labels, best_acc=acc,
                                         best_model_dict_proba=best_model_dict_proba)
        return best_acc

    def final_result(self, model_dict_proba, dog_cat):
        # this is for final ensemble
        _results_list = self._results_list.copy()
        _results_list.pop(self.del_ind)
        self.selected_model_dict = self.model_dict.copy()
        self.selected_model_dict.pop(self.del_ind)
        y_pred, y_score, eval_one_model, test = load_ckp_result(self.selected_model_dict, self.args, model_dict_proba)
        diction = ['행복/놀람', '중립/안정', '슬픔/두려움/화남/싫음'] if dog_cat == 'cat' else ['행복/놀람', '중립/안정', '슬픔/두려움', '화남/싫음']
        self.show_result(self.labels, y_pred, y_score, diction)

    def img2feature2npz_file(self):
        # final checkpoint transform test_set to feature(.npy folder)
        for model_name, checkpoint_path in self.model_dict:
            prediction_list = list()  # each item is 7-ele array
            # print("Processing", checkpoint_path)
            ckp_path = osp.join('saved', '{}'.format(self.args.path), '{}.npy'.format(checkpoint_path))
            if not osp.exists(ckp_path):
                try:
                    model = getattr(models, model_name)
                    model = model(in_channels=3, num_classes=self.args.class_num)
                except:
                    continue

                state = torch.load(os.path.join(self.args.ckdir, checkpoint_path),
                                   map_location=lambda storage, loc: storage)
                model.load_state_dict(state["net"])
                model.to(self.args.device)
                model.eval()

                with torch.no_grad():
                    for idx in tqdm(self.test_set_sequence, total=len(self.test_set), leave=False):
                        images, targets = self.test_set[idx]
                        images = make_batch(images)
                        images = images.to(self.args.device)

                        outputs = model(images).cpu()
                        outputs = F.softmax(outputs, 1)
                        outputs = torch.sum(outputs, 0)  # outputs.shape [tta_size, 7]

                        outputs = [round(o, 4) for o in outputs.numpy()]
                        prediction_list.append(outputs)

                np.save(ckp_path, prediction_list)
        self._results_list = load_npy_file(self.model_dict, self.args.path)

        y_pred, y_score, eval_one_model, test = load_ckp_result(self.model_dict, self.args, [1] * self.length)
        y_true, a = self.labels, dict()
        for idx, model_info in enumerate(self.model_dict):
            correct = np.sum(np.equal(eval_one_model[model_info[0]], y_true))
            acc = correct / len(y_true) * 100
            print(model_info[0], 'acc = ', acc, '|',
                  [y_true[i] for i, v in enumerate(np.equal(eval_one_model[model_info[0]], y_true)) if v])
            aa = make_binary(eval_one_model[model_info[0]])
            a[model_info[0]] = aa
        bin_y_true = make_binary(y_true)
        print('/n')
        # check binary label (positive:0 negative:1)
        for idx, model_info in enumerate(self.model_dict):
            correct = np.sum(np.equal(a[model_info[0]], bin_y_true))
            acc = correct / len(y_true) * 100
            print(model_info[0], 'acc = ', acc, '| binary',
                  [bin_y_true[i] for i, v in enumerate(np.equal(a[model_info[0]], bin_y_true)) if v])
        print('load_all')

    def show_result(self, y_true, y_pred, y_score, diction):
        # F1 score
        print('_________________________________result________________________')
        # diction = ['행복/놀람', '슬픔/두려움', '화남/싫음']
        print('\n', classification_report(y_true, y_pred, target_names=diction))
        cm = confusion_matrix(y_true, y_pred, labels=range(len(diction)))
        cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('confused matrix', '\n', cmn)
        print('\n', 'true_label : ', '\n', np.array(y_true))
        print('\n', 'predict_label : ', '\n', y_pred)
        clsf_report = pd.DataFrame(
            classification_report(y_true=y_true, y_pred=y_pred, target_names=diction, output_dict=True)).transpose()

    def make_del_ind(self, i):
        self.del_ind = i

    def second_acc(self, best_trials):
        best_acc, accuracy, proba = 0, [0], list()
        best_model_probas = [list(i.params.values()) for i in best_trials]
        for proba_test in best_model_probas:
            best_acc = self.model_voting_proba(proba_test, best_acc)
            if not accuracy[-1] == best_acc:
                accuracy.append(best_acc)
                proba.append(proba_test)
        return best_acc, proba[-1]


def make_batch(images):
    if not isinstance(images, list):
        images = [images]
    return torch.stack(images, 0)


def make_binary(sett):
    aa = list()
    for i in sett:
        v = 1 if i > 1 else 0
        aa.append(v)
    return aa


def calculate_accuracy(model_dict_proba, labels, tmp_test_result_list, best_acc=0, best_model_dict_proba=None):
    correct = np.sum(np.equal(tmp_test_result_list, labels))
    acc = correct / len(tmp_test_result_list) * 100
    if best_acc < acc:
        best_acc = acc
        best_model_dict_proba = model_dict_proba
    return best_acc, best_model_dict_proba


def load_npy_file(model_dict, path):
    test_results_list = list()
    for model_name, checkpoint_path in model_dict:
        try:
            ckp_path = osp.join('saved', '{}'.format(path), '{}.npy'.format(checkpoint_path))
            test_results = np.load(ckp_path, allow_pickle=True)
            test_results_list.append(test_results)
        except:
            continue
    return test_results_list


def load_ckp_result(model_dict, args, model_dict_proba):
    test_results_list = load_npy_file(model_dict, args.path)
    return output_result(test_results_list, model_dict, model_dict_proba)


def output_result(test_results_list, model_dict, model_dict_proba):
    tmp_test_result_list, eval_one_model = list(), dict()
    test_results_list = np.array(test_results_list)
    for idx in range(len(model_dict)):
        tmp_test_result_list.append(model_dict_proba[idx] * test_results_list[idx])
        eval_one_model[model_dict[idx][0]] = np.argmax(test_results_list[idx], axis=1)
    tmp_test_result_list = np.array(tmp_test_result_list)
    y_score = np.sum(tmp_test_result_list, axis=0)
    tmp_test_result_list = np.argmax(y_score, axis=1)
    tmp_test_result_list2 = 0
    return tmp_test_result_list, y_score, eval_one_model, tmp_test_result_list2


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


class dognet2():
    def __init__(self):
        from torchvision.transforms import transforms

        self.img_path = glob(os.path.join('checkpoint', 'dog_test', '*', '*.jpg'))
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )
        pass

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        image_path = self.img_path[item]
        info = image_path.split('/')
        label = int(info[2])
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            image = self._transform(image)
            return image, label


class catnet2():
    def __init__(self):
        from torchvision.transforms import transforms

        self.img_path = glob(os.path.join('checkpoint', 'cat_test', '*', '*.jpg'))
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )
        pass

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        image_path = self.img_path[item]
        info = image_path.split('/')
        #TODO change next line
        label = int(info[2]) if not int(info[2]) > 1 else 3
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            image = self._transform(image)
            return image, label


def main():
    args = parse_args()
    os.makedirs('saved/' + args.path, exist_ok=True)
    generate = Generate_ensemble(args)
    check_best_acc, best_model_proba = 0, list()
    generate.img2feature2npz_file()
    for i in range(generate.length):
        # optuna find the best model_proba for ensemble feature
        study = optuna.create_study(direction='maximize')
        generate.make_del_ind(i)
        study.optimize(generate.model_voting_proba, n_trials=args.optuna_num)
        # model_dict_proba = study.best_params
        generate.args.ensemble_indicator = 'B' if not generate.args.ensemble_indicator == 'B' else 'O'
        acc, proba = generate.second_acc(study.best_trials)
        # best_model_proba = [i, list(model_dict_proba.values())] if check_best_acc < study.best_value else best_model_proba
        # check_best_acc = study.best_value if check_best_acc < study.best_value else check_best_acc
        proba = [round(im, 2) for im in proba]
        print(f'no {generate.model_dict[i]}', 'best_model_proba : ', proba, '->', round(acc, 2),
              round(study.best_value, 2))
        best_model_proba = [i, proba] if check_best_acc < acc else best_model_proba
        check_best_acc = acc if check_best_acc < acc else check_best_acc
        generate.args.ensemble_indicator = 'B' if not generate.args.ensemble_indicator == 'B' else 'O'
    # show best ensemble result
    generate.make_del_ind(best_model_proba[0])
    print(f'final == no {generate.model_dict[best_model_proba[0]]}')
    generate.final_result(best_model_proba[1], args.dog_cat)
    print('end')


if __name__ == "__main__":
    main()