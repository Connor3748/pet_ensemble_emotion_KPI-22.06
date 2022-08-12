# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from glob import glob

import cv2
import mmcv
import numpy as np
import torch
from mmaction.detect_tool import Dogs

from mmaction.utils import import_module_error_func

# sys.path.append("../")
try:
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import (init_pose_model, inference_top_down_pose_model, vis_pose_result)
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
    parser.add_argument('--data_path', default=osp.join('..', 'dataset', 'dog'), help='data file path')
    parser.add_argument(
        '--checkpoint',
        default=('https://download.openmmlab.com/mmaction/skeleton/posec3d/'
                 'slowonly_r50_u48_240e_ntu120_xsub_keypoint/'
                 'slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth'),
        help='skeleton model checkpoint file/url')
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
    parser.add_argument('--det-score-thr', type=float, default=0.4, help='the threshold of dog detection score')
    parser.add_argument('--device', type=str, default='cuda:2', help='CPU/CUDA device option')
    parser.add_argument('--output_size', type=int, default=256, help='img_size')
    args = parser.parse_args()
    return args


def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.
    Returns:
        list[np.ndarray]: The dog detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[16] == 'dog', ('We require you to use a detector ''trained on COCO')
    results, dog_file_name = [], []
    print('Performing face Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)[16]
        check = result[:, 4] >= args.det_score_thr
        if any(check):
            dog_file_name.append(frame_path)
            results.append(result[check])
        prog_bar.update()
    # print((len(results) / 10), '% detect dog')
    return results, dog_file_name


def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
    results, dog_file_name, jump, labels, Dtypes = [], [], 1, [7], [6, 7]
    print('Performing Pose Estimation for each image')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for i, (f, d) in enumerate(zip(frame_paths, det_results)):
        prog_bar.update()
        datatype = f.split('/')[-2]
        label = labelfind(f)
        if jump:
            jump = jump - 1 if Dtypes[-1] == datatype else 0
            continue
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        len_eyes = pose[0]['keypoints'][0, 0] - pose[0]['keypoints'][1, 0]
        eyes_upper_than_nose = pose[0]['keypoints'][1, 0] < pose[0]['keypoints'][4, 0] < pose[0]['keypoints'][0, 0]
        eye_conf = all(pose[0]['keypoints'][0:2, 2] > 0.7)

        if eye_conf and len_eyes > 0 and eyes_upper_than_nose:  # xmin<xmean<xmax and ymin<ymean<ymax and
            results.append(pose)
            dog_file_name.append(f)
            labels.append(label)
            Dtypes.append(datatype)
            jump = data_inbalance(datatype, Dtypes, label)
    print((len(results) / 100), '% pose detect')
    return results, dog_file_name


def data_inbalance(datatype, Dtypes, label):
    jump = 6
    if label > 2:
        jump = 2
    elif label > 3:
        jump = 1
    if Dtypes[-3] == datatype:
        jump = jump * 4
    elif Dtypes[-2] == datatype:
        jump = jump * 2
    return jump


def labelfind(image_path):
    import codecs
    import json
    diction = ['행복/즐거움', '편안/안정', '불안/슬픔', '화남/불쾌', '공포', '공격성']
    label_path = os.path.join('../dataset/dog', image_path.split('/')[-3], image_path.split('/')[-2] + '.json')
    if os.path.exists(label_path) and os.path.exists(image_path):
        f = codecs.open(label_path, 'r')
        data = json.load(f)
        label = diction.index(data['metadata']['inspect']['emotion'])
        return label


def dog_img_check(path):
    imgs, paths = [], []
    prog_bar = mmcv.ProgressBar(len(path))
    for name in path:
        img = cv2.imread(name)
        imgs.append(img)
        paths.append(name)
        prog_bar.update()
    return paths, imgs


def facecrop(p, img_raw, img_size):
    (rx, ry), (lx, ly) = p[0]['keypoints'][0:2, 0:2]
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
    batch = 10000
    paths = glob(osp.join(args.data_path, 'Validation', '*', '**')) + glob(osp.join(args.data_path, 'Training', '*', '**'))

    for ind in range(0, len(paths), batch):
        print(f'now : {int(ind / batch)}/{int(len(paths) / batch)}')
        path = paths[ind:ind + min(batch, len(paths) - ind)]
        frame_paths, original_frames = dog_img_check(path)

        det_results, frame_paths = detection_inference(args, frame_paths)
        torch.cuda.empty_cache()

        pose_results, frame_paths = pose_inference(args, frame_paths, det_results)
        torch.cuda.empty_cache()

        count, check, img_size = 0, [], (args.output_size, args.output_size)
        for f, p in zip(frame_paths, pose_results):
            label = labelfind(f)
            img_raw = cv2.imread(f, cv2.IMREAD_COLOR)
            img = facecrop(p, img_raw, img_size)
            if f.split('/')[-2] in check:
                count = 0
                check.append(f.split('/')[-2])
            else:
                count = count + 1
            name = f.split('/')[-3].lower()[0:3] + '/' + f.split('/')[-2] + f'__{count}_{label}'
            cv2.imwrite(osp.join('cropdata', 'final', f'{name}.jpg'), img)
        print(len(pose_results))


if __name__ == '__main__':
    main()
