import os
import json
import numpy as np
import shutil

from cruw import CRUW


def load_json(json_path):
    with open(json_path) as f:
        json_dict = json.load(f)
    return json_dict


def get_frame_start_end(frame_id_center, win_size, max_frame_id):
    frame_id_center = int(frame_id_center)
    frame_id_start = frame_id_center - int(win_size / 2)
    frame_id_end = frame_id_center + int(win_size / 2)

    if frame_id_start < 0:
        frame_id_start = 0
        frame_id_end = 0 + win_size
    if frame_id_end > max_frame_id:
        frame_id_end = max_frame_id
        frame_id_start = frame_id_end - win_size

    return frame_id_start, frame_id_end


def process_camera_data(raw_root, out_root, split_name, seq_name, frame_id_start, frame_id_end):
    for fid, frame_id in enumerate(range(frame_id_start, frame_id_end)):
        img_path = os.path.join(raw_root, seq_name, 'camera', 'left', '%06d.png' % frame_id)
        if os.path.exists(img_path):
            os.makedirs(os.path.join(out_root, split_name, 'camera', 'left'), exist_ok=True)
            img_path_new = os.path.join(out_root, split_name, 'camera', 'left', '%s_%04d.png' % (seq_name, frame_id))
            shutil.copyfile(img_path, img_path_new)
        else:
            print('Error: %s, image file not found!' % img_path)


def process_radar_data(raw_root, out_root, split_name, seq_name, frame_id_start, frame_id_end,
                       radar_dim, chirp_ids):
    # process frame and chirp data into one numpy array
    radar_win = np.zeros(radar_dim, dtype=np.float32)
    for fid, frame_id in enumerate(range(frame_id_start, frame_id_end)):
        for cid, chirp_id in enumerate(chirp_ids):
            npy_path = os.path.join(raw_root, seq_name, 'radar', 'npy', 'ra',
                                    '%06d_%04d.npy' % (frame_id, chirp_id))
            if os.path.exists(npy_path):
                radar_win[fid, cid] = np.load(npy_path)
            else:
                print('Error: %s, npy file not found!' % npy_path)

    # save numpy file
    os.makedirs(os.path.join(out_root, split_name, 'radar'), exist_ok=True)
    npy_save_path = os.path.join(out_root, split_name, 'radar',
                                 '%s_%04d-%04d.npy' % (seq_name, frame_id_start, frame_id_end))
    np.save(npy_save_path, radar_win)


def process_label(raw_root, out_root, split_name, seq_name, frame_id_start, frame_id_end, cruw):
    label_convert_win = []
    for fid, frame_id in enumerate(range(frame_id_start, frame_id_end)):
        label_path = os.path.join(raw_root, seq_name, 'label', '%06d.json' % frame_id)
        label_convert_frame = []
        if os.path.exists(label_path):
            label_dicts = load_json(label_path)
            for label_dict in label_dicts:
                label_convert = {
                    'obj_type': label_dict['obj_type'],
                    'loc3d': {  # with coordinate translation
                        'x': -label_dict['psr']['position']['y'] + cruw.dataset.calib_cfg['t_rad2lid'][0],
                        'y': -label_dict['psr']['position']['z'] + cruw.dataset.calib_cfg['t_rad2lid'][1],
                        'z': label_dict['psr']['position']['x'] + cruw.dataset.calib_cfg['t_rad2lid'][2]
                    },
                    'dim3d': {
                        'l': label_dict['psr']['scale']['x'],
                        'w': label_dict['psr']['scale']['y'],
                        'h': label_dict['psr']['scale']['z']
                    }
                }  # TODO: add coordinate translation
                label_convert_frame.append(label_convert)

        else:  # no label file or no object in this frame
            pass

        label_convert_win.append(label_convert_frame)

    os.makedirs(os.path.join(out_root, split_name, 'label'), exist_ok=True)
    label_save_path = os.path.join(out_root, split_name, 'label',
                                   '%s_%04d-%04d.json' % (seq_name, frame_id_start, frame_id_end))
    with open(label_save_path, 'w') as f:
        json.dump(label_convert_win, f, indent=2)


def process_split(raw_root, out_root, split_name, split_frames, cruw, win_size=16, max_frame_id=1799):
    chirp_ids = cruw.sensor_cfg.radar_cfg['chirp_ids']
    n_chirps = len(chirp_ids)
    ramap_rsize = cruw.sensor_cfg.radar_cfg['ramap_rsize']
    ramap_asize = cruw.sensor_cfg.radar_cfg['ramap_asize']

    for split_frame in split_frames:
        print('processing %s ...' % split_frame)
        seq_name, frame_id_center = split_frame.split('/')
        frame_id_start, frame_id_end = get_frame_start_end(frame_id_center, win_size, max_frame_id)

        process_camera_data(raw_root, out_root, split_name, seq_name, frame_id_start, frame_id_end)

        radar_dim = (win_size, n_chirps, ramap_rsize, ramap_asize, 2)
        process_radar_data(raw_root, out_root, split_name, seq_name, frame_id_start, frame_id_end,
                           radar_dim, chirp_ids)

        process_label(raw_root, out_root, split_name, seq_name, frame_id_start, frame_id_end, cruw)


if __name__ == '__main__':
    raw_root = '/mnt/disk2/CRUW_2022'
    out_root = '/mnt/disk2/CRUW_2022_3DDet'
    splits = ['train']
    chirp_ids = [0, 64, 128, 192]
    json_path = os.path.join(os.getcwd(), '3ddet_cam_rad_03-13-2022.json')

    cruw = CRUW(raw_root, sensor_config_name='sensor_config_rod2021')  # TODO: add new sensor config for CRUW_2022
    split_dict = load_json(json_path)
    for split in splits:
        process_split(raw_root, out_root, split, split_dict[split], cruw)
