import random
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


model_dict = [
    ("resnet152", "resnet152_test_res152mixup_0224_2022Feb24_06.01"),
    ("resnet152", "resnet152_test_resnet152mixup_0225_2022Feb25_03.52"),
    ("inception_v3_", "inception_v3_test_inception_v3mixup_0225_2022Feb25_05.00"),
    ("resmasking", "resmasking_test_resmaskingmixup_0224_2022Feb24_06.00"),
    ("resmasking", "resmasking_test_resmaskmixup_0225_2022Feb25_01.12"),
    ("cbam_resnet50", "cbam_resnet50_test_cbam_resnet50_0304_2022Mar04_06.15"),

    ("densenet161", "densenet161_test_densenet161_0305_2022Mar07_00.38"),
    ("densenet161", "densenet161_test_densenet161mixup_0225_2022Feb25_01.09"),
    ("resmasking", "resmasking_test_resmasking_0307_2022Mar07_11.48"),
    ("resmasking", "resmasking_test_remasking_2022Feb11_05.24"),

    ("resmasking", "resmasking_test_resmasking_0225_2022Feb25_09.01"), ##

    ("resmasking", "resmasking_test_resmasking_0304_2022Mar04_06.12"),
    ("resmasking", "resmasking_test_resmasking_0305_2022Mar07_00.35"),
    ("inception_v3", "inception_v3_test_inception_v30225_2022Feb25_08.58"),
    ("inception_v3", "inception_v3_test_inception_v3_0304_2022Mar04_06.22"),
    ("inception_v3", "inception_v3_test_inception_v3_0305_2022Mar07_00.34"),
    ("inception_v3", "inception_v3_test_inception_v3_0307_2022Mar07_11.49"),
    ("resnet152", "resnet152_test_resnet152_0225_2022Feb25_09.00"),

    ("resnet152", "resnet152_test_resnet152_0305_2022Mar07_00.33"), ##

    ("resnet152", "resnet152_test_resnet152_0307_2022Mar07_11.52"),
    ("resnet152", "resnet152_test_resnet152pretrained_0221_2022Feb21_05.39"),
    ("resnet152", "resnet152_test_resnet152pretrained_2022Feb11_05.44"),
    ("densenet161", "densenet161_test_densenet161_0225_2022Feb25_09.09"),


    ("densenet161", "densenet161_test_densenet161mixup_0224_2022Feb24_06.51"),
    ("cbam_resnet50", "cbam_resnet50_test_cbam_resnet50_0307_2022Mar07_11.49"), ##
    ("cbam_resnet50", "cbam_resnet50_test_cbam_2022Feb11_02.00"), ##
    ("cbam_resnet50", "cbam_resnet50_test_cbam_resnet50_0225_2022Feb25_09.08"), ##
    ("cbam_resnet50", "cbam_resnet50_test_cbam_resnet50_0305_2022Mar07_00.36"), ##
    ("cbam_resnet50", "cbam_resnet50_test_cbam_resnet50mixup_0225_2022Feb25_01.48"),
]

# model_dict_proba_list = list(map(list, product([0, 1], repeat=len(model_dict))))


from glob import glob
import os
import codecs, json

def main():
    test_results_list = []

    basepath = '../../../'
    path = basepath + 'dataset/dogface/testll'
    img_paths = glob(os.path.join(path, "val/*.jpg")) + glob(os.path.join(path, "tra/*.jpg"))
    label_path = basepath + 'dataset/dog/Validation'
    diction = ['행복/즐거움', '편안/안정', '불안/슬픔', '화남/불쾌', '공포', '공격성']
    labels = []
    for image_path in img_paths:

        labelpath = os.path.join(label_path, image_path.split('/')[-1].split('__')[0] + '.json')
        if image_path.split('/')[-2] == 'val':
            labelpath = labelpath.replace('Training', 'Validation')
        if image_path.split('/')[-2] == 'tra':
            labelpath = labelpath.replace('Validation', 'Training')
        if os.path.exists(labelpath) and os.path.exists(image_path):
            f = codecs.open(labelpath, 'r')
            data = json.load(f)
            label = diction.index(data['metadata']['inspect']['emotion'])
            labels.append(label)

    for model_name, checkpoint_path in model_dict:
        test_results = np.load("../saved/results/{}.npy".format(checkpoint_path), allow_pickle=True)
        test_results_list.append(test_results)
    test_results_list = np.array(test_results_list)
    # load test targets
    #######
    #model_dict_proba = [0, 0, 1, 1, 0.5, 0.6, 0, 0, 0, 0]
    #model_dict_proba = [1,1,1,1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #model_dict_proba = [0.8, 2.5, 1.5, 0, 1, 1]
    for ii in range(29):
        model_dict_proba = [0]*ii + [1]

        tmp_test_result_list = []
        print(model_dict[ii][1])
        for idx in range(len(model_dict_proba)):
            tmp_test_result_list.append(model_dict_proba[idx] * test_results_list[idx])
        tmp_test_result_list = np.array(tmp_test_result_list)
        tmp_test_result_list = np.sum(tmp_test_result_list, axis=0)
        tmp_test_result_list = np.argmax(tmp_test_result_list, axis=1)

        correct = np.sum(np.equal(tmp_test_result_list, labels))

        acc = correct / len(tmp_test_result_list) * 100
        print(acc)

        labelcount, labelper = {}, {}
        for i in range(len(diction)):
            check, checkall = [], []
            for pr, tr in zip(tmp_test_result_list, labels):
                if not i == tr:
                    continue
                checkall.append(tr)
                if tr == pr:
                    check.append(tr)
            labelper[i] = int(len(check)/(len(checkall)+0.001) * 100)
            labelcount[i] = len(checkall)
        print(labelper)
    print((list(labelper.values())[0]*2.2+list(labelper.values())[1]-list(labelper.values())[2]*0.3+list(labelper.values())[3]*1.5+sum(list(labelper.values())[4:6])*0.3)/5)
    print('')
if __name__ == "__main__":
    main()
