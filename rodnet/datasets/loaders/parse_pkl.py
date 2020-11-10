import os


def list_pkl_filenames(dataset_configs, split):
    data_root = dataset_configs['data_root']
    seqs = dataset_configs[split]['seqs']
    seqs_pkl_names = [name + '.pkl' for name in seqs]
    return seqs_pkl_names
