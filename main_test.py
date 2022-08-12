# It is just for testing each ckpoint_model in check_path
# trash code
import codecs
import json
from collections import Counter
from glob import glob

import cv2
from torchvision.transforms import transforms

from classification import models
from classification.models import *

Result_EMOTION_DICT = {
    0: "행복/놀람(행복/즐거움)",
    1: "중립(편안/안정)",
    2: "두려움/슬픔(불안/슬픔/공포)",
    3: "화남/싫음(화남/불쾌/공격성)",
}

use_diction = ['행복/즐거움', '편안/안정', '불안/슬픔', '화남/불쾌', '공포', '공격성']


def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image

transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
is_cuda = torch.cuda.is_available()

# load configs and set random seed
package_root_dir = os.path.dirname(__file__)
configs = json.load(open("./configs/dog_config.json"))
check_path = glob('./checkpoint/cat_back/*cat*')

def main():
    # load configs and set random seed
    image_size = (configs["image_size"], configs["image_size"])
    device= 'cuda:2'
    configs["cwd"] = os.getcwd()
    # load model and data_loader
    for ind in check_path:
        try:
            _model = models.__dict__[ind.split('__')[0].split('/')[-1]]
            state = torch.load(f"{ind}", map_location=device)
            model = _model(in_channels=state['in_channels'], num_classes=state['num_classes'])

            model.load_state_dict(state["net"])
            model.to(device)
        except:
            print('============================', ind.split('__')[0], ' this checkpoint need to try again.. it is not checkpoint')
            continue
        model.eval()
        true, predict = list(), list()
        path = './cropdata/final6/testset'
        test_img = glob(os.path.join(path, "*.jpg"))
        label_paths = '../dataset/dog/Validation' if not 'cat' in path else '../dataset/cat/Validation'
        for i in test_img:
            if 'cat' in path:
                label_file = image_path.split('/')[-1].split('__')[0]
                label_path = os.path.join(label_paths, label_file.split('-')[1].upper(), label_file + '.json')
            else:
                labelpath = os.path.join(label_paths, i.split('/')[-1].split('__')[0] + '.json')
            if i.split('/')[-2] == 'tra':
                labelpath = labelpath.replace('Validation', 'Training')
            if os.path.exists(labelpath):
                f = codecs.open(labelpath, 'r')
                data = json.load(f)
                true_label = use_diction.index(data['metadata']['inspect']['emotion'])
                if true_label > 3:
                    true_label = 2 if true_label == 4 else 3
                if 'cat' in path:
                    if true_label > 1:
                        true_label = 2
                # if not (true_label == 4 or true_label == 5):
                #     continue
                true.append(Result_EMOTION_DICT[true_label])
            face_image = cv2.imread(i)
            assert isinstance(face_image, np.ndarray)
            face_images = ensure_color(face_image)
            face_images = cv2.resize(face_images, image_size)

            face_images = transform(face_images)
            if is_cuda:
                face_images = face_images.to(device)

            face_images = torch.unsqueeze(face_images, dim=0)

            output = torch.squeeze(model(face_images), 0)
            proba = torch.softmax(output, 0)

            # get dominant emotion
            emo_proba, emo_idx = torch.max(proba, dim=0)
            emo_idx = emo_idx.item()
            emo_label = Result_EMOTION_DICT[emo_idx]
            predict.append(emo_label)

        # get proba for each emotion
        labelcount, labelper = {}, {}
        for i in Result_EMOTION_DICT.values():
            check, checkall = [], []
            for tr, pr in zip(true, predict):
                if not i == tr:
                    continue
                checkall.append(tr)
                if tr == pr:
                    check.append(pr)
            labelcount[i] = len(checkall)
            labelper[i] = int(len(check) / (len(checkall)) * 100)
        print(ind)
        print(labelper)
        print('')


if __name__ == "__main__":
    main()