import os
import sys

from importlib import import_module


def load_configs_from_file(config_path):
    module_name = os.path.basename(config_path)[:-3]
    if '.' in module_name:
        raise ValueError('Dots are not allowed in config file path.')
    config_dir = os.path.dirname(config_path)
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {
        name: value
        for name, value in mod.__dict__.items()
        if not name.startswith('__')
    }
    return cfg_dict


def parse_cfgs(parser):
    # dataset_cfg
    parser.add_argument('--data_root', type=str,
                        help='directory to the dataset (will overwrite data_root in config file)')

    # model_cfg
    parser.add_argument('--model_type', type=str, help='model type')
    parser.add_argument('--model_name', type=str, help='model name or exp name')
    parser.add_argument('--max_dets', type=int, help='max detection per frome')
    parser.add_argument('--peak_thres', type=float, help='peak threshold')
    parser.add_argument('--ols_thres', type=float, help='OLS thres')
    parser.add_argument('--stacked_num', type=int, help='number of stack for HG')
    parser.add_argument('--mnet_cfg', type=tuple, help='MNet configuration')
    parser.add_argument('--dcn', type=bool, help='whether use TDC')

    # train_cfg
    parser.add_argument('--n_epoch', type=int, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--lr_step', type=int, help='step for learning rate decreasing')
    parser.add_argument('--win_size', type=int, help='window size for RF images')
    parser.add_argument('--train_step', type=int, help='training step within RF snippets')
    parser.add_argument('--train_stride', type=int, help='training stride between RF snippets')
    parser.add_argument('--log_step', type=int, help='step for printing out log info')
    parser.add_argument('--save_step', type=int, help='step for saving checkpoints')

    # test_cfg
    parser.add_argument('--test_step', type=int, help='testing step within RF snippets')
    parser.add_argument('--test_stride', type=int, help='testing stride between RF snippets')
    parser.add_argument('--rr_min', type=float, help='range of range min value')
    parser.add_argument('--rr_max', type=float, help='range of range max value')
    parser.add_argument('--ra_min', type=float, help='range of angle min value')
    parser.add_argument('--ra_max', type=float, help='range of angle max value')

    return parser


def update_config_dict(config_dict, args):
    # dataset_cfg
    if hasattr(args, 'data_root') and args.data_root is not None:
        data_root_old = config_dict['dataset_cfg']['base_root']
        config_dict['dataset_cfg']['base_root'] = args.data_root
        config_dict['dataset_cfg']['data_root'] = config_dict['dataset_cfg']['data_root'].replace(data_root_old,
                                                                                                  args.data_root)
        config_dict['dataset_cfg']['anno_root'] = config_dict['dataset_cfg']['anno_root'].replace(data_root_old,
                                                                                                  args.data_root)

    # model_cfg
    if hasattr(args, 'model_type') and args.model_type is not None:
        config_dict['model_cfg']['type'] = args.model_type
    if hasattr(args, 'model_name') and args.model_name is not None:
        config_dict['model_cfg']['name'] = args.model_name
    if hasattr(args, 'max_dets') and args.max_dets is not None:
        config_dict['model_cfg']['max_dets'] = args.max_dets
    if hasattr(args, 'peak_thres') and args.peak_thres is not None:
        config_dict['model_cfg']['peak_thres'] = args.peak_thres
    if hasattr(args, 'ols_thres') and args.ols_thres is not None:
        config_dict['model_cfg']['ols_thres'] = args.ols_thres
    if hasattr(args, 'stacked_num') and args.stacked_num is not None:
        config_dict['model_cfg']['stacked_num'] = args.stacked_num
    if hasattr(args, 'mnet_cfg') and args.mnet_cfg is not None:
        config_dict['model_cfg']['mnet_cfg'] = args.mnet_cfg
    if hasattr(args, 'dcn') and args.dcn is not None:
        config_dict['model_cfg']['dcn'] = args.dcn

    # train_cfg
    if hasattr(args, 'n_epoch') and args.n_epoch is not None:
        config_dict['train_cfg']['n_epoch'] = args.n_epoch
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config_dict['train_cfg']['batch_size'] = args.batch_size
    if hasattr(args, 'lr') and args.lr is not None:
        config_dict['train_cfg']['lr'] = args.lr
    if hasattr(args, 'lr_step') and args.lr_step is not None:
        config_dict['train_cfg']['lr_step'] = args.lr_step
    if hasattr(args, 'win_size') and args.win_size is not None:
        config_dict['train_cfg']['win_size'] = args.win_size
    if hasattr(args, 'train_step') and args.train_step is not None:
        config_dict['train_cfg']['train_step'] = args.train_step
    if hasattr(args, 'train_stride') and args.train_stride is not None:
        config_dict['train_cfg']['train_stride'] = args.train_stride
    if hasattr(args, 'log_step') and args.log_step is not None:
        config_dict['train_cfg']['log_step'] = args.log_step
    if hasattr(args, 'save_step') and args.save_step is not None:
        config_dict['train_cfg']['save_step'] = args.save_step

    # test_cfg
    if hasattr(args, 'test_step') and args.test_step is not None:
        config_dict['test_cfg']['test_step'] = args.test_step
    if hasattr(args, 'test_stride') and args.test_stride is not None:
        config_dict['test_cfg']['test_stride'] = args.test_stride
    if hasattr(args, 'rr_min') and args.rr_min is not None:
        config_dict['test_cfg']['rr_min'] = args.rr_min
    if hasattr(args, 'rr_max') and args.rr_max is not None:
        config_dict['test_cfg']['rr_max'] = args.rr_max
    if hasattr(args, 'ra_min') and args.ra_min is not None:
        config_dict['test_cfg']['ra_min'] = args.ra_min
    if hasattr(args, 'ra_max') and args.ra_max is not None:
        config_dict['test_cfg']['ra_max'] = args.ra_max

    return config_dict
