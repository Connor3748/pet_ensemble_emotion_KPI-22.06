# Copyright (c) OpenMMLab. All rights reserved.
# this is for crop_dog_face using gpu
'''
This code is for crop_face
well.. this code
1) make a lot of ckpt model [OK] [main_dog.py] -> [./checkpoint]
2) use model ensemble [OK] [main_test_2022.py]
3) but! this code make best voting rate for test_set to use KOIST test using optuna library [def model_voting_proba(self, trial)]
4) so if you choose same voting rate to another real data, it doesn't probably work well. I guess

* if you wanna use real, just change "list(model_dict_proba.values())" to some list
( that list equal to the number of model... like [1, 1, 1, 1] if use 4 model ensemble )
'''

import argparse
import codecs
import json
import os
import os.path as osp
from glob import glob

import cv2
import mmcv
import numpy as np
import torch

from mmaction.utils import import_module_error_func

# sys.path.append("../")
try:
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import (init_pose_model, inference_top_down_pose_model, vis_pose_result)
    from mmpose.datasets.pipelines import Compose
    from mmcv.parallel import collate
except (ImportError, ModuleNotFoundError):
    @import_module_error_func('mmdet')
    def inference_detector(*args, **kwargs):
        pass


    @import_module_error_func('mmdet')
    def init_detector(*args, **kwargs):
        pass


    @import_module_error_func('mmpose')
    def init_pose_model(*args, **kwargs):
        pass


    @import_module_error_func('mmpose')
    def inference_top_down_pose_model(*args, **kwargs):
        pass


    @import_module_error_func('mmpose')
    def vis_pose_result(*args, **kwargs):
        pass
try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument('--config',
                        default=('configs/skeleton/posec3d/''slowonly_r50_u48_240e_ntu120_xsub_keypoint.py'),
                        help='skeleton model config file path')
    parser.add_argument('--data_path', default=osp.join('..', 'dataset', 'cat'), help='data file path')
    parser.add_argument('--det-config', default='demo/yolox_x_8x8_300e_coco.py',
                        help='Dog detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default='https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
        , help='dog detection checkpoint file/url')
    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_animalpose_256x256.py',
        help='dog pose estimation config file path (from mmpose)')
    parser.add_argument(
        '--pose-checkpoint',
        default='https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth'
        , help='dog pose estimation checkpoint file/url')
    parser.add_argument('--det-score-thr', type=float, default=0.3, help='the threshold of dog detection score')
    parser.add_argument('--device', type=str, default='cuda:2', help='CPU/CUDA device option')
    parser.add_argument('--output_size', type=int, default=256, help='img_size')
    parser.add_argument('--dog_cat', default='cat', help='choose dog or cat')
    parser.add_argument('--save_path', default='cropdata/cat4', help='path like dog1 or dog2 dog3 dog4 is to save crop face img')
    args = parser.parse_args()
    return args


def data_inbalance(datatype, Dtypes, label):
    # this function is just for .mp4 data's problem
    jump = 8 if not label == 1 else 16
    if not label == 1:
        jump = 1 if Dtypes[-2] != datatype else 2
    # if label == 2:
    #     jump = 1 if Dtypes[-2] != datatype else 2
    # elif label == 3:
    #     jump = 0 if Dtypes[-2] != datatype else 1
    if Dtypes[-2] == datatype:
        jump = jump * 4
    elif Dtypes[-1] == datatype:
        jump = jump * 2
    return jump


def labelfind(image_path, args):
    diction = ['행복/즐거움', '편안/안정', '불안/슬픔', '화남/불쾌', '공포', '공격성']
    label_path = os.path.join('../dataset', args.dog_cat, image_path.split('/')[-3],
                              image_path.split('/')[-2] + '.json')
    if os.path.exists(label_path) and os.path.exists(image_path):
        f = codecs.open(label_path, 'r')
        data = json.load(f)
        label = diction.index(data['metadata']['inspect']['emotion'])
        return label


def facecrop(p, img_raw, img_size):
    (rx, ry), (lx, ly) = p[0:2, 0:2]
    center = ((lx + rx) // 2, (ly + ry) // 2)
    angle = np.degrees(np.arctan2(ry - ly, rx - lx))
    scale = 61.44 / (np.sqrt(((rx - lx) ** 2) + ((ry - ly) ** 2)))
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += (img_size[0] / 2 - center[0])
    M[1, 2] += (img_size[1] / 2 + int(img_size[1] / 20) - center[1])
    img = cv2.warpAffine(img_raw, M, img_size, borderValue=0.0)
    return img


def main():
    args = parse_args()
    os.makedirs(osp.join(args.save_path, 'tra'), exist_ok=True)
    os.makedirs(osp.join(args.save_path, 'val'), exist_ok=True)
    batch, check_size, img_size = 200, 10000, args.output_size
    if args.data_path == osp.join('..', 'dataset', 'dog'):
        paths = glob(osp.join(args.data_path, 'Validation', '*', '**')) + glob(
            osp.join(args.data_path, 'Training', '*', '**'))
    else:
        paths = glob(osp.join(args.data_path, 'Validation', '*', '**', '*.jpg')) + glob(
            osp.join(args.data_path, 'Training', '*', '**', '*.jpg'))
    detect_model = init_detector(args.det_config, args.det_checkpoint, args.device)
    det_label = 16 if args.dog_cat == 'dog' else 15
    assert detect_model.CLASSES[det_label] == args.dog_cat, ('We require you to use a detector ''trained on COCO')
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
    cfg = pose_model.cfg
    device = next(pose_model.parameters()).device
    channel_order = cfg.test_pipeline[0].get('channel_order', 'rgb')
    test_pipeline = [LoadImage(channel_order=channel_order)] + cfg.test_pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    flip_pairs = [[0, 1], [2, 3], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19]]

    checkfile, jump = [10, 9], 0
    for ind in range(0, len(paths), batch):
        dataname, label, pathss, img_stack, img_metas_stack, stack_box = list(), list(), list(), list(), list(), list()
        if ind % check_size == 0:
            print(f'now : {int(ind / check_size)}/{int(len(paths) / check_size)}',
                  (100 * (len(checkfile) - 2) / check_size), '% pose detect')
            checkfile, jump, count = [10, 9], 0, 0
            prog_bar = mmcv.ProgressBar(int(len(paths) / check_size / 2) + 1)

        batch_img = paths[ind:ind + min(batch, len(paths) - ind)]
        det_result = np.array(inference_detector(detect_model, batch_img))[:, det_label].tolist()

        prog_bar.update()
        for i, det in enumerate(det_result):
            if not len(det):
                pass
            else:
                true_false = det[:, 4] > args.det_score_thr
                path = batch_img[i]

                if not any(true_false):
                    continue
                pathss.append(path)
                bboxes_xyxy = det[true_false][0]
                stack_box.append(bboxes_xyxy)
                bbox_xywh = bboxes_xyxy.copy()
                bbox_xywh[2] = bbox_xywh[2] - bbox_xywh[0] + 1
                bbox_xywh[3] = bbox_xywh[3] - bbox_xywh[1] + 1
                dataname.append(path.split('/')[-2]), label.append(labelfind(path, args))
                if label[-1] == 3 :
                    continue
                center, scale = _box2cs(cfg, bbox_xywh)
                # prepare data
                data = {'img_or_path': path, 'center': center, 'scale': scale,
                        'bbox_score': bboxes_xyxy[4] if len(bboxes_xyxy) == 5 else 1,
                        'bbox_id': 0,  # need to be assigned if batch_size > 1
                        'dataset': 'animalpose',
                        'joints_3d': np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
                        'joints_3d_visible': np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32), 'rotation': 0,
                        'ann_info': {'image_size': np.array(cfg.data_cfg['image_size']),
                                     'num_joints': cfg.data_cfg['num_joints'], 'flip_pairs': flip_pairs}}
                data = test_pipeline(data)
                img_stack.append(data['img'].to(device))
                img_metas_stack.append(data['img_metas'].data)
        img_stack = collate(img_stack, samples_per_gpu=1)
        with torch.no_grad():
            results = pose_model(img=img_stack, img_metas=img_metas_stack, return_loss=False)['preds']
        # height, width = path.shape[:2]
        # person_results = [{'bbox': np.array([0, 0, width, height])}]
        for i, result in enumerate(results):
            if jump:
                jump = jump - 1 if checkfile[-1] == dataname else 0
                continue
            left_right_eyes = result[0, 0] - result[1, 0]
            nose_between_eyes = result[1, 0] < result[4, 0] < result[0, 0]
            eye_conf = all(result[0:2, 2] > 0.7)
            if eye_conf and left_right_eyes > 0 and nose_between_eyes:
                jump = data_inbalance(dataname[i], checkfile, label[i])
                img_raw = cv2.imread(pathss[i], cv2.IMREAD_COLOR)
                img = facecrop(result, img_raw, (img_size, img_size))
                count = count + 1 if dataname[i] in checkfile else 0
                if args.data_path == osp.join('..', 'dataset', 'dog'):
                    name = osp.join(pathss[i].split('/')[-3].lower()[0:3], dataname[i] + f'__{count}')
                else:
                    name = osp.join(pathss[i].split('/')[-4].lower()[0:3], dataname[i] + f'__{count}')
                cv2.imwrite(osp.join(args.save_path, f'{name}.jpg'), img)
                checkfile.append(dataname[i])


class LoadImage:
    """A simple pipeline to load image."""

    def __init__(self, color_type='color', channel_order='rgb'):
        self.color_type = color_type
        self.channel_order = channel_order

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the img_or_path.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img_or_path'], str):
            results['image_file'] = results['img_or_path']
            img = mmcv.imread(results['img_or_path'], self.color_type,
                              self.channel_order)
        elif isinstance(results['img_or_path'], np.ndarray):
            results['image_file'] = ''
            if self.color_type == 'color' and self.channel_order == 'rgb':
                img = cv2.cvtColor(results['img_or_path'], cv2.COLOR_BGR2RGB)
            else:
                img = results['img_or_path']
        else:
            raise TypeError('"img_or_path" must be a numpy array or a str or '
                            'a pathlib.Path object')

        results['img'] = img
        return results


def _box2cs(cfg, box):
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    input_size = cfg.data_cfg['image_size']
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

    scale = scale * 1.25

    return center, scale


if __name__ == '__main__':
    main()
