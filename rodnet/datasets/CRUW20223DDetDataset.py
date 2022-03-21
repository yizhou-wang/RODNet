import os
import json
import time
import random
import numpy as np
from tqdm import tqdm

import torch
import torchvision.transforms as T
from torch.utils import data

from cruw.mapping.coor_transform import cart2pol_ramap
from cruw.mapping.ops import ra2idx_interpolate
from cruw.mapping.object_types import get_class_id

from rodnet.datasets.transforms import normalize
from rodnet.utils.image import gaussian_radius, draw_msra_gaussian, draw_umich_gaussian


class CRUW20223DDetDataset(data.Dataset):
    """
    Pytorch Dataloader for CR Dataset
    :param detail_dir: data details directory
    :param confmap_dir: confidence maps directory
    :param win_size: seqence window size
    :param n_class: number of classes for detection
    :param step: frame step inside each sequence
    :param stride: data sampling
    :param set_type: train, valid, test
    :param is_random_chirp: random load chirp or not
    """

    def __init__(self, data_dir, dataset, config_dict, split, is_random_chirp=True,
                 transform=None, noise_channel=False, old_normalize=False):
        # parameters settings
        self.data_dir = data_dir
        self.dataset = dataset
        self.config_dict = config_dict
        self.n_class = dataset.object_cfg.n_class
        self.win_size = config_dict['train_cfg']['win_size']
        self.split = split
        if split == 'train' or split == 'valid':
            self.step = config_dict['train_cfg']['train_step']
            self.stride = config_dict['train_cfg']['train_stride']
        else:
            self.step = config_dict['test_cfg']['test_step']
            self.stride = config_dict['test_cfg']['test_stride']
        self.is_random_chirp = is_random_chirp
        self.n_chirps = 1
        self.transform = transform
        self.noise_channel = noise_channel

        # Dataloader for MNet
        if 'mnet_cfg' in self.config_dict['model_cfg']:
            in_chirps, out_channels = self.config_dict['model_cfg']['mnet_cfg']
            self.n_chirps = in_chirps
        self.chirp_ids = self.dataset.sensor_cfg.radar_cfg['chirp_ids']

        # dataset initialization
        self.radar_paths = self.get_radar_image_paths()
        self.obj_infos = self.get_labels()
        self.image_paths = self.get_camera_image_paths()
        self.n_data = len(self.radar_paths)

        self.old_normalize = old_normalize
        self.mean = torch.tensor([-0.0990, -0.9608])
        self.std = torch.tensor([531.9800, 531.0670])

    def __len__(self):
        """Total number of data/label pairs"""
        return self.n_data

    def __getitem__(self, index):
        image_paths = self.image_paths[index]
        radar_path = self.radar_paths[index]
        obj_info_win = self.obj_infos[index]
        radar_configs = self.dataset.sensor_cfg.radar_cfg

        data_dict = dict(
            status=True,
            image_paths=image_paths,
            radar_path=radar_path
        )

        # Load radar data
        radar_win = np.load(radar_path)
        radar_win = self.transform_radar_data(radar_win, old_normalize=self.old_normalize)
        data_dict['radar_data'] = radar_win

        # Load annotations
        if self.split == 'train' or self.split == 'valid':
            confmap_gt = self.generate_confmap(obj_info_win)
            if self.noise_channel:
                assert confmap_gt.shape == \
                       (self.n_class + 1, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
            else:
                confmap_gt = confmap_gt[:self.n_class]
                assert confmap_gt.shape == \
                       (self.n_class, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
            data_dict['anno'] = dict(
                obj_infos=obj_info_win,
                confmaps=confmap_gt,
            )
        else:
            data_dict['anno'] = None

        return data_dict

    def get_radar_image_paths(self):
        radar_data_dir = os.path.join(self.data_dir, self.split, self.dataset.sensor_cfg.radar_cfg['chirp_folder'])
        radar_data_names = os.listdir(radar_data_dir)
        radar_data_names.sort()
        radar_data_paths = [os.path.join(radar_data_dir, fname) for fname in radar_data_names]
        return radar_data_paths

    def get_labels(self):
        label_paths = [path.replace('.npy', '.json').replace(self.dataset.sensor_cfg.radar_cfg['chirp_folder'], 'label')
                       for path in self.radar_paths]
        labels = []
        for label_path in label_paths:
            if os.path.exists(label_path):
                with open(label_path) as f:
                    labels_frame = json.load(f)
            else:
                if self.split == 'train' or self.split == 'valid':
                    assert FileNotFoundError, "label file is not found %s" % label_path
                else:
                    labels_frame = []
            labels.append(labels_frame)
        return labels

    def get_camera_image_paths(self):
        camera_data_dir = os.path.join(self.data_dir, self.split, self.dataset.sensor_cfg.camera_cfg['image_folder'])
        camera_paths = []
        for radar_path in self.radar_paths:
            camera_paths_ = []
            radar_name = radar_path.split('/')[-1]
            seq_name, frame_ids = self.get_seq_frame_id_from_filename(radar_name)
            for frame_id in frame_ids:
                img_name = '%s_%04d.%s' % (seq_name, frame_id, self.dataset.sensor_cfg.camera_cfg['ext'])
                img_path = os.path.join(camera_data_dir, img_name)
                assert os.path.exists(img_path)
                camera_paths_.append(img_path)
            camera_paths.append(camera_paths_)
        return camera_paths

    def get_seq_frame_id_from_filename(self, filename):
        filename = filename.split('.')[0]
        year, date, seq, frames = filename.split('_')
        seq_name = '_'.join([year, date, seq])
        frame_start, frame_end = frames.split('-')
        frame_start, frame_end = int(frame_start), int(frame_end)
        frame_ids = list(range(frame_start, frame_end))
        return seq_name, frame_ids

    def transform_radar_data(self, radar_npy, old_normalize=False):
        radar_configs = self.dataset.sensor_cfg.radar_cfg
        radar_tensor = torch.from_numpy(radar_npy)
        radar_tensor = radar_tensor.view([-1, radar_configs['ramap_rsize'], radar_configs['ramap_asize'], 2])
        radar_tensor = radar_tensor.permute(0, 3, 1, 2)
        if old_normalize:
            radar_tensor /= 3e+04
        else:
            normalize(radar_tensor, mean=self.mean, std=self.std, inplace=True)
        if 'mnet_cfg' in self.config_dict['model_cfg']:
            radar_tensor = radar_tensor.view([self.win_size, self.n_chirps, 2,
                                              radar_configs['ramap_rsize'],
                                              radar_configs['ramap_asize']])
            radar_tensor = radar_tensor.permute(2, 0, 1, 3, 4)
        else:
            radar_tensor = radar_tensor.view([self.win_size, 2,
                                              radar_configs['ramap_rsize'],
                                              radar_configs['ramap_asize']])
            radar_tensor = radar_tensor.permute(1, 0, 2, 3)

        # radar_npy_win = torch.nn.functional.normalize(radar_npy_win, dim=-1)

        # radar_npy_win_amp = np.sqrt(radar_npy_win[..., 0] ** 2 + radar_npy_win[..., 1] ** 2)
        # radar_npy_win_amp = radar_npy_win_amp[..., np.newaxis]
        # radar_npy_win_amp = np.repeat(radar_npy_win_amp, 2, axis=-1)
        # radar_npy_win[radar_npy_win_amp > 1] /= radar_npy_win_amp[radar_npy_win_amp > 1]
        # print('max: %.4f, min: %.4f' % (radar_npy_win.max(), radar_npy_win.min()))
        # print('max: %.4f, min: %.4f' % (radar_npy_win_amp.max(), radar_npy_win_amp.min()))

        return radar_tensor

    def generate_confmap(self, obj_info_win):
        draw_gaussian = draw_msra_gaussian if self.config_dict['model_cfg']['loss'] == 'mse' else draw_umich_gaussian
        hm_win = np.zeros((self.config_dict['model_cfg']['n_class_train'],
                           self.config_dict['train_cfg']['win_size'],
                           self.dataset.sensor_cfg.radar_cfg['ramap_asize'],
                           self.dataset.sensor_cfg.radar_cfg['ramap_rsize']), dtype=np.float32)
        for win_id in range(self.config_dict['train_cfg']['win_size']):
            labels = obj_info_win[win_id]
            num_objs = len(labels)
            hm = np.zeros((self.config_dict['model_cfg']['n_class_train'],
                           self.dataset.sensor_cfg.radar_cfg['ramap_asize'],
                           self.dataset.sensor_cfg.radar_cfg['ramap_rsize']), dtype=np.float32)

            for k in range(num_objs):
                ct, cls_id, w = self.convert_ann_to_grid(labels[k])
                if w > 0:
                    radius = 2 * gaussian_radius((w, w))
                    radius = max(10, int(radius))
                    radius = 10 if self.config_dict['model_cfg']['loss'] == 'mse' else radius
                    radius = int(np.ceil(radius))
                    draw_gaussian(hm[cls_id], ct, radius)
            hm_win[:, win_id, :, :] = hm
        return hm_win

    def convert_ann_to_grid(self, ann_dict):
        x = ann_dict['loc3d']['x']
        z = ann_dict['loc3d']['z']
        rng, agl = cart2pol_ramap(x, z)
        rng_id, agl_id = ra2idx_interpolate(rng, agl, self.dataset.range_grid, self.dataset.angle_grid)
        rng_id = int(np.round(rng_id))
        agl_id = int(np.round(agl_id))
        ct = [agl_id, rng_id]

        if type(ann_dict['obj_type']) == str:
            cls_id = get_class_id(ann_dict['obj_type'].lower(), self.dataset.object_cfg.classes)
        else:
            print('wrong annotation:', ann_dict)
            return ct, 0, -1
        if cls_id < 0:
            return ct, 0, -1

        if rng >= self.dataset.range_grid[-1]:
            # outside range of radar
            return ct, cls_id, -1

        rrw = max(ann_dict['dim3d']['l'], ann_dict['dim3d']['w'])
        half_width_view_angle = np.arcsin(rrw / 2 / rng)
        rid1, aid1 = ra2idx_interpolate(rng, agl - half_width_view_angle,
                                        self.dataset.range_grid,
                                        self.dataset.angle_grid)
        rid2, aid2 = ra2idx_interpolate(rng, agl + half_width_view_angle,
                                        self.dataset.range_grid,
                                        self.dataset.angle_grid)
        rrw_pixel = aid2 - aid1
        return ct, cls_id, rrw_pixel
