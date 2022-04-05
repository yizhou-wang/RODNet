import os
import argparse
import numpy as np
import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from cruw.cruw import CRUW
from cruw.visualization.draw_rf import magnitude
from cruw.mapping import rf2rfcart


def parse_args():
    parser = argparse.ArgumentParser(description="convert RF image from ra to cart coordinates")

    parser.add_argument('--data_root', default='/mnt/nas_cruw/CRUW_2022', type=str,
                        help='data root path')
    parser.add_argument('--split', default='train', type=str,
                        help='dataset split')

    parser.add_argument('--xz_dim', default=(200, 151), type=tuple,
                        help='radar RF image xz coordinates dimension')
    parser.add_argument('--zrange', default=30.0, type=float,
                        help='largest range for z axis')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    dataset = CRUW(data_root=args.data_root, sensor_config_name='./configs/dataset_configs/sensor_config_cruw2022.json')

    seq_names = sorted(os.listdir(args.data_root))
    seq_names = [s for s in seq_names if len(s) == 14]
    # seq_names = seq_names[9:10]

    for seq_name in seq_names:
        frame_dir = os.path.join(args.data_root, seq_name, dataset.sensor_cfg.radar_cfg['chirp_folder'])
        frame_names = os.listdir(frame_dir)
        frame_names_sel = []
        for frame_name in frame_names:
            chirp_id = int(frame_name.split('.')[0].split('_')[1])
            if chirp_id in dataset.sensor_cfg.radar_cfg['chirp_ids']:
                frame_names_sel.append(frame_name)
        n_frame = len(frame_names_sel)  # / dataset.sensor_cfg.radar_cfg['n_chirps']

        # load sensor calibration params
        xz_grid = dataset.xz_grid

        # preload camera image paths and radar images
        print("Pre-loading camera and radar images %s ..." % seq_name)
        for frameid in tqdm.tqdm(range(n_frame)):
            chirp_npy_dir = os.path.join(args.data_root, seq_name, 'radar/npy/ra_cart')
            os.makedirs(chirp_npy_dir, exist_ok=True)
            chirp_npy_path = os.path.join(chirp_npy_dir, frame_names_sel[frameid])

            chirp_vis_dir = os.path.join(args.data_root, seq_name, 'radar/vis/ra_cart')
            os.makedirs(chirp_vis_dir, exist_ok=True)
            chirp_vis_path = os.path.join(chirp_vis_dir, frame_names_sel[frameid].replace('.npy', '.jpg'))

            if os.path.exists(chirp_npy_path) and os.path.exists(chirp_vis_path):
                # skip if files exist
                continue

            # save npy files
            chirp_data = np.load(os.path.join(frame_dir, frame_names_sel[frameid]))
            chirp_data_cart, _ = rf2rfcart(chirp_data, dataset.range_grid, dataset.angle_grid,
                                           xz_grid, magnitude_only=False)
            np.save(chirp_npy_path, chirp_data_cart)

            # visualization
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax_im = ax.imshow(magnitude(chirp_data_cart, radar_data_type='RI'), origin='lower', vmin=0, vmax=5000)
            fig.colorbar(ax_im)
            ax.set_xticks(np.arange(0, len(xz_grid[0]), 30), xz_grid[0][::30])
            ax.set_yticks(np.arange(0, len(xz_grid[1]), 20), xz_grid[1][::20])
            ax.set_xlabel('x(m)')
            ax.set_ylabel('z(m)')
            fig.savefig(chirp_vis_path)
            fig.clf()
            plt.close(fig)
