import os
import time
import random
import pickle
import numpy as np
from tqdm import tqdm

from torch.utils import data

from .collate_functions import _cr_collate_npy
from .loaders import list_pkl_filenames, list_pkl_filenames_from_prepared


class CRDatasetSM(data.Dataset):
    """
    Pytorch Dataloader for CR Dataset
    :param detail_dir: data details directory
    :param confmap_dir: confidence maps directory
    :param win_size: seqence window size
    :param n_class: number of classes for detection
    :param step: frame step inside each sequence
    :param stride: data sampling
    :param set_type: train, valid, test
    :param is_random: random load or not
    """

    def __init__(self, data_dir, dataset, config_dict, split, is_random=True, subset=None, noise_channel=False):
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
        self.is_random = is_random
        self.n_chirps = 1
        self.noise_channel = noise_channel

        # Dataloader for MNet
        if 'mnet_cfg' in self.config_dict['model_cfg']:
            in_chirps, out_channels = self.config_dict['model_cfg']['mnet_cfg']
            self.n_chirps = in_chirps
        self.chirp_ids = self.dataset.sensor_cfg.radar_cfg['chirp_ids']

        # dataset initialization
        self.image_paths = []
        self.radar_paths = []
        self.obj_infos = []
        self.n_data = 0
        self.index_mapping = []

        if subset is not None:
            self.data_files = [subset + '.pkl']
        else:
            # self.data_files = list_pkl_filenames(config_dict['dataset_cfg'], split)
            self.data_files = list_pkl_filenames_from_prepared(data_dir, split)
        self.seq_names = [name.split('.')[0] for name in self.data_files]
        self.n_seq = len(self.seq_names)

        for seq_id, data_file in enumerate(tqdm(self.data_files)):
            data_file_path = os.path.join(data_dir, split, data_file)
            data_details = pickle.load(open(data_file_path, 'rb'))
            if split == 'train' or split == 'valid':
                assert data_details['anno'] is not None
            n_frame = data_details['n_frame']
            self.image_paths.append(data_details['image_paths'])
            self.radar_paths.append(data_details['radar_paths'])
            n_data_in_seq = (n_frame - (self.win_size * self.step - 1)) // self.stride + (
                1 if (n_frame - (self.win_size * self.step - 1)) % self.stride > 0 else 0)
            self.n_data += n_data_in_seq
            for data_id in range(n_data_in_seq):
                self.index_mapping.append([seq_id, data_id * self.stride])
            if data_details['anno'] is not None:
                self.obj_infos.append(data_details['anno']['metadata'])

    def __len__(self):
        """Total number of data/label pairs"""
        return self.n_data

    def __getitem__(self, index):

        seq_id, data_id = self.index_mapping[index]
        image_paths = self.image_paths[seq_id]
        radar_paths = self.radar_paths[seq_id]
        this_data_file = os.path.join(self.data_dir, self.split, self.data_files[seq_id])
        this_data_details = pickle.load(open(this_data_file, 'rb'))
        if this_data_details['anno'] is not None:
            this_seq_obj_info = this_data_details['anno']['metadata']
            this_seq_confmap = this_data_details['anno']['confmaps']

        data_dict = dict(
            status=True,
            image_paths=[]
        )

        if self.is_random:
            chirp_id = random.randint(0, len(self.chirp_ids) - 1)
        else:
            chirp_id = 0

        # Dataloader for MNet
        if 'mnet_cfg' in self.config_dict['model_cfg']:
            chirp_id = self.chirp_ids

        radar_configs = self.dataset.sensor_cfg.radar_cfg
        ramap_rsize = radar_configs['ramap_rsize']
        ramap_asize = radar_configs['ramap_asize']

        # Load radar data
        try:
            if radar_configs['data_type'] == 'RI' or radar_configs['data_type'] == 'AP':  # drop this format
                radar_npy_win = np.load(radar_paths[chirp_id]) \
                    [data_id:data_id + self.win_size * self.step:self.step, :, :, :]
                for idx, frameid in enumerate(
                        range(data_id, data_id + self.win_size * self.step, self.step)):
                    data_dict['image_paths'].append(image_paths[frameid])
            elif radar_configs['data_type'] == 'RISEP' or radar_configs['data_type'] == 'APSEP':
                if isinstance(chirp_id, int):
                    radar_npy_win = np.zeros((self.win_size, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                    for idx, frameid in enumerate(
                            range(data_id, data_id + self.win_size * self.step, self.step)):
                        radar_npy_win[idx, :, :, :] = np.load(radar_paths[frameid][chirp_id])
                        data_dict['image_paths'].append(image_paths[frameid])
                elif isinstance(chirp_id, list):
                    radar_npy_win = np.zeros((self.win_size, self.n_chirps, ramap_rsize, ramap_asize, 2),
                                             dtype=np.float32)
                    for idx, frameid in enumerate(
                            range(data_id, data_id + self.win_size * self.step, self.step)):
                        for cid, c in enumerate(chirp_id):
                            npy_path = radar_paths[frameid][c]
                            radar_npy_win[idx, cid, :, :, :] = np.load(npy_path)
                        data_dict['image_paths'].append(image_paths[frameid])
                else:
                    raise TypeError
            elif radar_configs['data_type'] == 'ROD2021':
                if isinstance(chirp_id, int):
                    radar_npy_win = np.zeros((self.win_size, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                    for idx, frameid in enumerate(
                            range(data_id, data_id + self.win_size * self.step, self.step)):
                        radar_npy_win[idx, :, :, :] = np.load(radar_paths[frameid][chirp_id])
                        data_dict['image_paths'].append(image_paths[frameid])
                elif isinstance(chirp_id, list):
                    radar_npy_win = np.zeros((self.win_size, self.n_chirps, ramap_rsize, ramap_asize, 2),
                                             dtype=np.float32)
                    for idx, frameid in enumerate(
                            range(data_id, data_id + self.win_size * self.step, self.step)):
                        for cid, c in enumerate(chirp_id):
                            npy_path = radar_paths[frameid][cid]
                            radar_npy_win[idx, cid, :, :, :] = np.load(npy_path)
                        data_dict['image_paths'].append(image_paths[frameid])
                else:
                    raise TypeError
            else:
                raise ValueError

            data_dict['start_frame'] = data_id
            data_dict['end_frame'] = data_id + self.win_size * self.step - 1

        except:
            # in case load npy fail
            data_dict['status'] = False
            if not os.path.exists('./tmp'):
                os.makedirs('./tmp')
            log_name = 'loadnpyfail-' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
            with open(os.path.join('./tmp', log_name), 'w') as f_log:
                f_log.write('npy path: ' + radar_paths[frameid][chirp_id] + \
                            '\nframe indices: %d:%d:%d' % (data_id, data_id + self.win_size * self.step, self.step))
            return data_dict

        # Dataloader for MNet
        if 'mnet_cfg' in self.config_dict['model_cfg']:
            radar_npy_win = np.transpose(radar_npy_win, (4, 0, 1, 2, 3))
            assert radar_npy_win.shape == (
                2, self.win_size, self.n_chirps, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
        else:
            radar_npy_win = np.transpose(radar_npy_win, (3, 0, 1, 2))
            assert radar_npy_win.shape == (2, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

        data_dict['radar_data'] = radar_npy_win

        # Load annotations
        if this_data_details['anno'] is not None:
            confmap_gt = this_seq_confmap[data_id:data_id + self.win_size * self.step:self.step]
            confmap_gt = np.transpose(confmap_gt, (1, 0, 2, 3))
            obj_info = this_seq_obj_info[data_id:data_id + self.win_size * self.step:self.step]
            if self.noise_channel:
                assert confmap_gt.shape == \
                       (self.n_class + 1, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
            else:
                confmap_gt = confmap_gt[:self.n_class]
                assert confmap_gt.shape == \
                       (self.n_class, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

            data_dict['anno'] = dict(
                obj_infos=obj_info,
                confmaps=confmap_gt,
            )
        else:
            data_dict['anno'] = None

        return data_dict

    def getBatch(self, indexes):
        """
        Get data batch with the given indices
        TODO: change return data format
        :param indexes:
        :return:
        """
        data_dicts = []
        for index in indexes:
            data_dict = self.__getitem__(index)
            data_dicts.append(data_dict)
        data_dict_batch = _cr_collate_npy(data_dicts)
        return data_dict_batch

    # def getObjInfo(self, index):
    #     seq_id, data_id = self.index_mapping[index]
    #     this_data_file = os.path.join(self.data_root, self.split, self.data_files[seq_id])
    #     this_data_details = pickle.load(open(this_data_file, 'rb'))
    #     obj_info = this_data_details['anno']['obj_infos'][data_id:data_id + self.win_size * self.step:self.step]
    #     return obj_info
    #
    # def getBatchObjInfo(self, indexes):
    #     bObj_info = []
    #     for index in indexes:
    #         result = self.getObjInfo(index)
    #         bObj_info.append(result)
    #     return bObj_info


if __name__ == "__main__":
    dataset = CRDatasetSM('./data/data_details', stride=16)
    print(len(dataset))
    for i in range(len(dataset)):
        continue
