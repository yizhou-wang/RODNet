import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


def create_model(config_dict, args, dataset, is_train=True):
    model_cfg = config_dict['model_cfg']
    print("Building model ... (%s)" % model_cfg)

    n_class = dataset.object_cfg.n_class
    if args.use_noise_channel:
        n_class_train = n_class + 1
    else:
        n_class_train = n_class
    config_dict['model_cfg']['n_class_train'] = n_class_train

    if 'stacked_num' in model_cfg:
        stacked_num = model_cfg['stacked_num']
    else:
        stacked_num = None
        config_dict['model_cfg']['stacked_num'] = stacked_num

    if model_cfg['type'] == 'CDC':
        from rodnet.models import RODNetCDC as RODNet
        rodnet = RODNet(in_channels=2, n_class=n_class_train).cuda()

    elif model_cfg['type'] == 'HG':
        from rodnet.models import RODNetHG as RODNet
        rodnet = RODNet(in_channels=2, n_class=n_class_train, stacked_num=stacked_num).cuda()

    elif model_cfg['type'] == 'HGwI':
        from rodnet.models import RODNetHGwI as RODNet
        rodnet = RODNet(in_channels=2, n_class=n_class_train, stacked_num=stacked_num).cuda()

    elif model_cfg['type'] == 'CDCv2':
        from rodnet.models import RODNetCDCDCN as RODNet
        in_chirps = len(dataset.sensor_cfg.radar_cfg['chirp_ids'])
        rodnet = RODNet(in_channels=in_chirps, n_class=n_class_train,
                        mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                        dcn=config_dict['model_cfg']['dcn']).cuda()

    elif model_cfg['type'] == 'HGv2':
        from rodnet.models import RODNetHGDCN as RODNet
        in_chirps = len(dataset.sensor_cfg.radar_cfg['chirp_ids'])
        rodnet = RODNet(in_channels=in_chirps, n_class=n_class_train, stacked_num=stacked_num,
                        mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                        dcn=config_dict['model_cfg']['dcn']).cuda()

    elif model_cfg['type'] == 'HGwIv2':
        from rodnet.models import RODNetHGwIDCN as RODNet
        in_chirps = len(dataset.sensor_cfg.radar_cfg['chirp_ids'])
        rodnet = RODNet(in_channels=in_chirps, n_class=n_class_train, stacked_num=stacked_num,
                        mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                        dcn=config_dict['model_cfg']['dcn']).cuda()

    else:
        raise NotImplementedError

    if is_train:
        if config_dict['model_cfg']['loss'] == 'mse':
            criterion = nn.MSELoss()
        elif config_dict['model_cfg']['loss'] == 'bce':
            criterion = nn.BCELoss()
        else:
            raise NotImplementedError

        optimizer = optim.Adam(rodnet.parameters(), lr=config_dict['train_cfg']['lr'])
        scheduler = StepLR(optimizer, step_size=config_dict['train_cfg']['lr_step'], gamma=0.1)

        return rodnet, criterion, optimizer, scheduler

    else:
        return rodnet


def load_checkpoint(rodnet, cp_path, optimizer=None, is_train=True):
    checkpoint = torch.load(cp_path)
    train_id_dict = {}
    loss_dict = {}
    if 'optimizer_state_dict' in checkpoint:
        rodnet.load_state_dict(checkpoint['model_state_dict'])
        if is_train and optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_id_dict['epoch_start'] = checkpoint['epoch'] + 1
        train_id_dict['iter_start'] = checkpoint['iter'] + 1
        loss_dict['loss_cp'] = checkpoint['loss']
        if 'iter_count' in checkpoint:
            train_id_dict['iter_count'] = checkpoint['iter_count']
        if 'loss_ave' in checkpoint:
            loss_dict['loss_ave'] = checkpoint['loss_ave']
    else:
        rodnet.load_state_dict(checkpoint)
    if is_train:
        return rodnet, optimizer, train_id_dict, loss_dict
    else:
        return rodnet


def save_model(model_name, epoch, iter, iter_count, rodnet, optimizer, loss_confmap, loss_ave, save_model_path):
    print("saving model %d/%d/%d ..." % (epoch + 1, iter + 1, iter_count + 1))
    status_dict = {
        'model_name': model_name,
        'epoch': epoch + 1,
        'iter': iter + 1,
        'model_state_dict': rodnet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_confmap.item(),
        'loss_ave': loss_ave,
        'iter_count': iter_count + 1,
    }
    torch.save(status_dict, save_model_path)
