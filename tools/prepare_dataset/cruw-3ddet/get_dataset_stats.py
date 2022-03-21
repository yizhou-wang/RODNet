import tqdm
import torch
import argparse

from cruw import CRUW
from rodnet.datasets.dataset_utils import get_dataloader
from rodnet.utils.load_configs import load_configs_from_file, parse_cfgs, update_config_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Train RODNet for CRUW 3DDet dataset.')

    parser.add_argument('--config', type=str, help='configuration file path')
    parser.add_argument('--sensor_config', type=str,
                        default='./configs/dataset_configs/sensor_config_cruw2022_3ddet.json')
    # parser.add_argument('--data_dir', type=str, default='./data/', help='directory to the prepared data')
    parser.add_argument('--log_dir', type=str, default='./checkpoints/', help='directory to save trained model')
    parser.add_argument('--resume_from', type=str, default=None, help='path to the trained model')
    parser.add_argument('--use_noise_channel', action="store_true", help="use noise channel or not")

    parser = parse_cfgs(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    config_dict = load_configs_from_file(args.config)
    config_dict = update_config_dict(config_dict, args)  # update configs by args

    dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], sensor_config_name=args.sensor_config)

    n_class = dataset.object_cfg.n_class
    if args.use_noise_channel:
        n_class_train = n_class + 1
    else:
        n_class_train = n_class
    config_dict['model_cfg']['n_class_train'] = n_class_train

    crdata_train, dataloader = get_dataloader(dataset.dataset, config_dict, args, dataset)

    # COMPUTE MEAN / STD
    # placeholders
    psum = torch.tensor([0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0])

    # loop through images
    for inputs in tqdm.tqdm(crdata_train):
        radar_tensor = inputs['radar_data']
        # radar_tensor = torch.from_numpy(inputs['radar_data'])
        psum += radar_tensor.sum(axis=(1, 2, 3, 4))
        psum_sq += (radar_tensor ** 2).sum(axis=(1, 2, 3, 4))

    # FINAL CALCULATIONS
    # pixel count
    count = len(crdata_train) * \
            config_dict['train_cfg']['win_size'] * \
            len(dataset.sensor_cfg.radar_cfg['chirp_ids']) * \
            dataset.sensor_cfg.radar_cfg['ramap_rsize'] * \
            dataset.sensor_cfg.radar_cfg['ramap_asize']

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq - total_mean ** 2) / count
    total_std = torch.sqrt(total_var)

    # output
    print('mean: ' + str(total_mean))
    print('std:  ' + str(total_std))
