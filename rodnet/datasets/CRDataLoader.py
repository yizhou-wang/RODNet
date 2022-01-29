import numpy as np
import random
import ctypes
import torch
from multiprocessing import Array, Process


class CRDataLoader:
    def __init__(self, dataset, shuffle=False, num_parallel_batch=2, noise_channel=False):
        # parameters settings
        self.dataset = dataset
        self.config_dict = self.dataset.config_dict
        self.n_class = dataset.dataset.object_cfg.n_class
        self.batch_size = self.config_dict['train_cfg']['batch_size']
        self.radar_configs = self.dataset.dataset.sensor_cfg.radar_cfg
        self.model_configs = self.config_dict['model_cfg']
        self.ramap_rsize = self.radar_configs['ramap_rsize']
        self.ramap_asize = self.radar_configs['ramap_asize']
        self.n_chirps = self.dataset.n_chirps

        if noise_channel:
            self.n_class = self.n_class + 1
        else:
            self.n_class = self.n_class

        self.length = len(dataset) // self.batch_size + (1 if len(dataset) % self.batch_size != 0 else 0)
        self.loading_seq = [i for i in range(len(dataset))]
        if shuffle:
            random.shuffle(self.loading_seq)
        self.restart = False

        assert num_parallel_batch > 0 and type(num_parallel_batch) == int

        self.win_size = dataset.win_size
        n_shradar = num_parallel_batch * self.batch_size * 2 * dataset.win_size * self.n_chirps * self.ramap_rsize \
                    * self.ramap_asize
        self.shradar = Array(ctypes.c_double, n_shradar)
        n_shconf = num_parallel_batch * self.batch_size * self.n_class * dataset.win_size * self.ramap_rsize \
                   * self.ramap_asize
        self.shconf = Array(ctypes.c_double, n_shconf)
        self.num_parallel_batch = num_parallel_batch

    def __len__(self):
        return self.length

    def __iter__(self):
        data_dict_stack = [None, None]
        procs = [None, None]

        random.shuffle(self.loading_seq)
        cur_loading_seq = self.loading_seq[:self.batch_size]
        data_dict_stack[0] = self.dataset.getBatch(cur_loading_seq)
        procs[0] = Process(target=self.getBatchArray,
                           args=(self.shradar, self.shconf, data_dict_stack[0], cur_loading_seq, 0))
        procs[0].start()

        index_num = self.num_parallel_batch - 1
        for i in range(self.__len__()):
            index_num = (index_num + 1) % self.num_parallel_batch
            procs[index_num].join()
            procs[index_num] = None

            if i < self.length - self.num_parallel_batch:
                cur_loading_seq = self.loading_seq[
                                  self.batch_size * (i + self.num_parallel_batch - 1): self.batch_size * (
                                          i + self.num_parallel_batch)]
            else:
                cur_loading_seq = self.loading_seq[self.batch_size * (i + self.num_parallel_batch - 1):]

            if i < self.length - self.num_parallel_batch + 1:
                stack_id_next = (index_num + 1) % self.num_parallel_batch
                data_dict_stack[stack_id_next] = self.dataset.getBatch(cur_loading_seq)
                procs[stack_id_next] = Process(target=self.getBatchArray,
                                               args=(self.shradar, self.shconf, data_dict_stack[stack_id_next],
                                                     cur_loading_seq, (index_num + 1) % self.num_parallel_batch))
                procs[stack_id_next].start()

            shradarnp = np.frombuffer(self.shradar.get_obj())
            if self.n_chirps == 1:
                shradarnp = shradarnp.reshape(self.num_parallel_batch, self.batch_size, 2, self.win_size,
                                              self.ramap_rsize, self.ramap_asize)
            else:
                shradarnp = shradarnp.reshape(self.num_parallel_batch, self.batch_size, 2, self.win_size,
                                              self.n_chirps, self.ramap_rsize, self.ramap_asize)
            shconfnp = np.frombuffer(self.shconf.get_obj())
            shconfnp = shconfnp.reshape(self.num_parallel_batch, self.batch_size, self.n_class, self.win_size,
                                        self.ramap_rsize, self.ramap_asize)

            if i < self.length - 1:
                data_length = self.batch_size
            else:
                data_length = len(self.dataset) - self.batch_size * i

            data_dict_return = dict(
                status=data_dict_stack[index_num]['status'],
                image_paths=data_dict_stack[index_num]['image_paths'],
                radar_data=torch.from_numpy(shradarnp[index_num, :data_length, :, :, :, :]),
                anno=dict(
                    obj_infos=data_dict_stack[index_num]['anno']['obj_infos'],
                    confmaps=torch.from_numpy(shconfnp[index_num, :data_length, :, :, :, :]),
                )
            )
            yield data_dict_return

    def __getitem__(self, index):
        if self.restart:
            random.shuffle(self.loading_seq)
        if index == self.length - 1:
            self.restart = True
            results = self.dataset.getBatch(self.loading_seq[self.batch_size * index:])
        else:
            results = self.dataset.getBatch(self.loading_seq[self.batch_size * index: self.batch_size * (index + 1)])
        results = list(results)
        for i in range(2):
            results[i] = torch.from_numpy(results[i])
        return results

    def getBatchArray(self, shradar, shconf, data_dict, loading_seq, index):
        shradarnp = np.frombuffer(shradar.get_obj())
        if self.n_chirps == 1:
            shradarnp = shradarnp.reshape(self.num_parallel_batch, self.batch_size, 2, self.win_size,
                                          self.ramap_rsize, self.ramap_asize)
        else:
            shradarnp = shradarnp.reshape(self.num_parallel_batch, self.batch_size, 2, self.win_size, self.n_chirps,
                                          self.ramap_rsize, self.ramap_asize)
        shconfnp = np.frombuffer(shconf.get_obj())
        shconfnp = shconfnp.reshape(self.num_parallel_batch, self.batch_size, self.n_class, self.win_size,
                                    self.ramap_rsize, self.ramap_asize)
        shradarnp[index, :len(loading_seq), :, :, :, :] = data_dict['radar_data']
        shconfnp[index, :len(loading_seq), :, :, :, :] = data_dict['anno']['confmaps']

    # def getBatchObjInfo(self, index):
    #     if index == self.length - 1:
    #         results = self.dataset.getBatchObjInfo(self.loading_seq[self.batch_size * index:])
    #     else:
    #         results = self.dataset.getBatchObjInfo(
    #             self.loading_seq[self.batch_size * index: self.batch_size * (index + 1)])
    #     return results
