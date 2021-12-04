import os
import argparse
import numpy as np

from cruw import CRUW
from cruw.eval import evaluate_rodnet_seq
from cruw.eval.rod.rod_eval_utils import accumulate, summarize

from rodnet.utils.load_configs import load_configs_from_file

olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
recThrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RODNet.')
    parser.add_argument('--data_root', type=str, default='./data/', help='directory to the prepared data')
    parser.add_argument('--sensor_config', type=str, default='sensor_config_rod2021')
    parser.add_argument('--gt_dir', type=str, default='./results/', help='directory to ground truth')
    parser.add_argument('--res_dir', type=str, default='./results/', help='directory to save testing results')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    Example:
        python eval.py --config configs/<CONFIG_FILE> --res_dir results/<FOLDER_NAME>
    """
    args = parse_args()
    dataset = CRUW(data_root=args.data_root, sensor_config_name=args.sensor_config)

    seq_names = sorted(os.listdir(args.res_dir))
    seq_names = [name for name in seq_names if '.' not in name]

    evalImgs_all = []
    n_frames_all = 0

    for seq_name in seq_names:
        gt_path = os.path.join(args.gt_dir, seq_name.upper() + '.txt')
        res_path = os.path.join(args.res_dir, seq_name, 'rod_res.txt')

        data_path = os.path.join(dataset.data_root, 'sequences', 'test', gt_path.split('/')[-1][:-4])
        n_frame = len(os.listdir(os.path.join(data_path, dataset.sensor_cfg.camera_cfg['image_folder'])))

        evalImgs = evaluate_rodnet_seq(res_path, gt_path, n_frame, dataset)
        eval = accumulate(evalImgs, n_frame, olsThrs, recThrs, dataset, log=False)
        stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
        print("%s | AP_total: %.4f | AR_total: %.4f" % (seq_name.upper(), stats[0] * 100, stats[1] * 100))

        n_frames_all += n_frame
        evalImgs_all.extend(evalImgs)

    eval = accumulate(evalImgs_all, n_frames_all, olsThrs, recThrs, dataset, log=False)
    stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
    print("%s | AP_total: %.4f | AR_total: %.4f" % ('Overall'.ljust(18), stats[0] * 100, stats[1] * 100))
