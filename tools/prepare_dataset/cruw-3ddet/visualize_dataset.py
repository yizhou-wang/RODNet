import os
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib import gridspec

from cruw import CRUW
from cruw.visualization.draw_rgb import draw_dets
from cruw.visualization.draw_rf import draw_centers
from cruw.visualization.utils import generate_colors_rgb
from cruw.mapping.ops import ra2idx, xz2raidx
from cruw.mapping.coor_transform import pol2cart_ramap, cart2pol_ramap


def load_json(json_path):
    with open(json_path) as f:
        json_dict = json.load(f)
    return json_dict


def get_sorted_filenames(folder_path):
    file_names = os.listdir(folder_path)
    file_names.sort()
    return file_names


def convert_lidar_to_radar(loc3d_lidar, cruw):
    x = loc3d_lidar['x']
    y = loc3d_lidar['y']
    z = loc3d_lidar['z']
    xyz_lid = np.array([x, y, z])
    t_rad2lid = np.array(cruw.sensor_cfg.calib_cfg['t_cl2lid']) - np.array(cruw.sensor_cfg.calib_cfg['t_cl2rad'])
    xyz_rad = xyz_lid + t_rad2lid
    return xyz_rad[0], xyz_rad[1], xyz_rad[2]


def draw_data_frame(image_path, chirp, anno_dict, save_path, cruw):
    img = plt.imread(image_path)

    fig = plt.figure()
    fig.set_size_inches(16, 5)
    gs = gridspec.GridSpec(1, 2)

    ax1 = plt.subplot(gs[0])
    ax1.axis('off')
    draw_dets(ax1, img, [], [])
    ax1.set_title('RGB Image')

    ax2 = plt.subplot(gs[1])
    ax2.axis('off')
    n_obj = len(anno_dict)
    categories = []
    center_ids = []
    for obj_id in range(n_obj):
        x, y, z = convert_lidar_to_radar(anno_dict[obj_id]['loc3d'], cruw)
        if z < cruw.range_grid[-1]:  # only show annotations with range grid
            categories.append(anno_dict[obj_id]['obj_type'])

            # use geometric center
            # rng_id, agl_id = xz2raidx(x, z, cruw.range_grid, cruw.angle_grid)

            # use nearest center
            rng, agl = cart2pol_ramap(x, z)
            rrw = max(anno_dict[obj_id]['dim3d']['l'], anno_dict[obj_id]['dim3d']['w'])
            rng -= rrw / 4
            rng_id, agl_id = ra2idx(rng, agl, cruw.range_grid, cruw.angle_grid)

            center_ids.append((rng_id, agl_id))

    colors = generate_colors_rgb(n_obj)
    draw_centers(ax2, chirp, center_ids, colors, texts=categories, normalized=False)
    ax2.set_title('RF Image')

    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(save_path)
    plt.close(fig)


def visualize_split(data_root, split, cruw):
    camera_dir = os.path.join(data_root, split, 'camera', 'left')
    radar_dir = os.path.join(data_root, split, 'radar')
    label_dir = os.path.join(data_root, split, 'label')
    vis_dir = os.path.join(data_root, split, 'vis')
    os.makedirs(vis_dir, exist_ok=True)

    camera_names = get_sorted_filenames(camera_dir)
    radar_names = get_sorted_filenames(radar_dir)
    label_names = get_sorted_filenames(label_dir)

    n_data = len(radar_names)
    for data_id in range(n_data):
        print('visualizing %s ...' % radar_names[data_id][:-4])
        year, date, seq, frames = radar_names[data_id][:-4].split('_')
        seq_name = '_'.join([year, date, seq])
        frame_id_start, frame_id_end = frames.split('-')
        frame_id_start, frame_id_end = int(frame_id_start), int(frame_id_end)
        radar_path = os.path.join(radar_dir, radar_names[data_id])
        radar_data_win = np.load(radar_path)
        win_size, n_chirps, ramap_rsize, ramap_asize, n_channels = radar_data_win.shape

        label_name = radar_names[data_id].replace('.npy', '.json')
        label_path = os.path.join(label_dir, label_name)
        label_dict_win = load_json(label_path)

        for win_id in range(win_size):
            camera_name = '%s_%04d.png' % (seq_name, frame_id_start + win_id)
            assert camera_name in camera_names
            camera_path = os.path.join(camera_dir, camera_name)

            save_path = os.path.join(vis_dir, camera_name)
            draw_data_frame(camera_path, radar_data_win[win_id, 0], label_dict_win[win_id], save_path, cruw)


if __name__ == '__main__':
    data_root = '/mnt/disk2/CRUW_2022_3DDet'
    splits = ['train']

    # cruw = CRUW(data_root, sensor_config_name='sensor_config_rod2021')  # TODO: add new sensor config for CRUW_2022
    config_path = '../../../configs/dataset_configs/sensor_config_cruw2022_3ddet.json'
    cruw = CRUW(data_root, sensor_config_name=config_path)

    for split in splits:
        visualize_split(data_root, split, cruw)
