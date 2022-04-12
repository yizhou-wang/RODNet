import os
import json
import argparse
import math
import numpy as np
from tqdm import tqdm

from cruw import CRUW
from cruw.eval import evaluate_rodnet_cruw2022
from cruw.eval.rod.load_txt import read_rodnet_res
from cruw.eval.rod.rod_eval_utils import accumulate, summarize
from cruw.mapping.coor_transform import cart2pol_ramap
from cruw.mapping.ops import ra2idx_interpolate
from cruw.mapping.object_types import get_class_id

from rodnet.datasets.CRUW2022Dataset import SPLIT_SEQ_DICT

olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
recThrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)

LOCAL_LABEL_DIR = '/mnt/disk2/CRUW_2022/CRUW_2022_label'
PART_SEQ_TRAINING = 0.7
USE_GEO_CENTER = False


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RODNet.')
    parser.add_argument('--data_root', type=str, help='directory to the prepared data')
    parser.add_argument('--sensor_config', type=str,
                        default='./configs/dataset_configs/sensor_config_cruw2022.json')
    parser.add_argument('--gt_dir', type=str, default=LOCAL_LABEL_DIR, help='directory to ground truth')
    parser.add_argument('--res_dir', type=str, help='directory to save testing results')
    args = parser.parse_args()
    return args


def convert_label(label_dict, frame_id, obj_id, dataset):
    label_convert = {
        'obj_type': label_dict['obj_type'],
        'loc3d': {
            'x': -label_dict['psr']['position']['y'],
            'y': -label_dict['psr']['position']['z'],
            'z': label_dict['psr']['position']['x']
        },
        'dim3d': {
            'l': label_dict['psr']['scale']['x'],
            'w': label_dict['psr']['scale']['y'],
            'h': label_dict['psr']['scale']['z']
        }
    }
    x = label_convert['loc3d']['x']
    z = label_convert['loc3d']['z']
    rng, agl = cart2pol_ramap(x, z)

    if USE_GEO_CENTER:
        # use geometric center
        # print('using geometric center')
        rng_id, agl_id = ra2idx_interpolate(rng, agl, dataset.range_grid, dataset.angle_grid)

    else:
        # use nearest center
        # print('using nearest center')
        rrw = max(label_convert['dim3d']['l'], label_convert['dim3d']['w'])
        rng -= rrw / 4
        rng_id, agl_id = ra2idx_interpolate(rng, agl, dataset.range_grid, dataset.angle_grid)

    rng_id = int(np.round(rng_id))
    agl_id = int(np.round(agl_id))

    if type(label_convert['obj_type']) == str:
        cls_id = get_class_id(label_convert['obj_type'].lower(), dataset.object_cfg.classes)
    else:
        print('wrong annotation:', label_convert)
        return None
    if cls_id < 0:
        return None

    if rng > 25 or rng < 1:
        return None
    if agl > math.radians(60) or agl < math.radians(-60):
        return None

    obj_dict = dict(
        id=obj_id,
        frame_id=frame_id,
        range=rng,
        angle=agl,
        range_id=rng_id,
        angle_id=agl_id,
        class_id=cls_id,
        score=1.0
    )

    return obj_dict


def get_labels(gt_dir, img_names, dataset):
    n_frame = len(img_names)
    n_class = dataset.object_cfg.n_class

    dts = {(i, j): [] for i in range(n_frame) for j in range(n_class)}
    obj_id = 1

    for frame_id, img_name in enumerate(img_names):  # frame_id starts from 0
        frame_id_real = img_name.split('.')[0]
        label_name = frame_id_real + '.json'
        label_path = os.path.join(gt_dir, label_name)
        # frame_id = int(frame_id)
        if os.path.exists(label_path):
            with open(label_path) as f:
                labels_frame = json.load(f)
        else:
            labels_frame = []
        for label_dict in labels_frame:
            label_convert = convert_label(label_dict, frame_id, obj_id, dataset)
            if label_convert is not None:
                dts[frame_id, label_convert['class_id']].append(label_convert)
            obj_id += 1

    return dts


if __name__ == '__main__':
    """
    Example:
        python eval.py --config configs/<CONFIG_FILE> --res_dir results/<FOLDER_NAME>
    """
    args = parse_args()
    dataset = CRUW(data_root=args.data_root, sensor_config_name=args.sensor_config)

    seq_names = sorted(SPLIT_SEQ_DICT['test'])

    evalImgs_all = []
    n_frames_all = 0

    for seq_name in seq_names:
        data_path = os.path.join(dataset.data_root, seq_name, dataset.sensor_cfg.camera_cfg['image_folder'])
        frame_names = os.listdir(data_path)
        n_frame = len(frame_names)
        n_frame_train = int(n_frame * PART_SEQ_TRAINING)
        frame_names = frame_names[n_frame_train:]
        n_frame = len(frame_names)

        res_path = os.path.join(args.res_dir, seq_name, 'rod_res.txt')
        res_dets = read_rodnet_res(res_path, n_frame, dataset)

        gt_dir = os.path.join(args.gt_dir, seq_name, 'label')
        gt_dets = get_labels(gt_dir, frame_names, dataset)

        evalImgs = evaluate_rodnet_cruw2022(res_dets, gt_dets, n_frame, dataset)
        eval = accumulate(evalImgs, n_frame, olsThrs, recThrs, dataset, log=False)
        stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
        print("%s | AP_total: %.4f | AR_total: %.4f" % (seq_name.upper(), stats[0] * 100, stats[1] * 100))

        n_frames_all += n_frame
        evalImgs_all.extend(evalImgs)

    eval = accumulate(evalImgs_all, n_frames_all, olsThrs, recThrs, dataset, log=False)
    stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
    print("%s | AP_total: %.4f | AR_total: %.4f" % ('Overall'.ljust(14), stats[0] * 100, stats[1] * 100))
