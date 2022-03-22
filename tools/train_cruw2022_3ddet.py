import os
import time
import argparse
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from cruw import CRUW

from rodnet.datasets.dataset_utils import get_dataloader
from rodnet.models.model_utils import create_model, load_checkpoint, save_model
from rodnet.trains.visualization import vis_train
from rodnet.trains.train_utils import save_train_configs
from rodnet.utils.solve_dir import create_dir_for_new_model
from rodnet.utils.load_configs import load_configs_from_file, parse_cfgs, update_config_dict


def parse_args():
    parser = argparse.ArgumentParser(description='Train RODNet for CRUW 3DDet dataset.')

    parser.add_argument('--config', type=str, help='configuration file path')
    parser.add_argument('--sensor_config', type=str,
                        default='./configs/dataset_configs/sensor_config_cruw2022_3ddet.json')
    # parser.add_argument('--data_dir', type=str, default='./data/', help='directory to the prepared data')
    parser.add_argument('--log_dir', type=str, default='./checkpoints/', help='directory to save trained model')
    parser.add_argument('--resume_from', type=str, default=None, help='path to the trained model')

    parser = parse_cfgs(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config_dict = load_configs_from_file(args.config)
    config_dict = update_config_dict(config_dict, args)  # update configs by args
    dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], sensor_config_name=args.sensor_config)

    # prepare create model
    cp_path = None
    epoch_start, iter_start, iter_count = 0, 0, 0
    loss_ave = 0.
    if args.resume_from is not None and os.path.exists(args.resume_from):
        cp_path = args.resume_from
        model_dir, model_name = create_dir_for_new_model(config_dict['model_cfg']['name'], args.log_dir)
    else:
        model_dir, model_name = create_dir_for_new_model(config_dict['model_cfg']['name'], args.log_dir)

    # create model
    rodnet, criterion, optimizer, scheduler = create_model(config_dict, args, dataset)

    # load model
    if cp_path is not None:
        print('loading pretrained model %s ...' % cp_path)
        rodnet, optimizer, train_id_dict, loss_dict = load_checkpoint(rodnet, args.resume_from, optimizer)

    # setup dataloader
    crdata_train, dataloader = get_dataloader(dataset.dataset, config_dict, args, dataset)
    try:
        seq_names = crdata_train.seq_names
        index_mapping = crdata_train.index_mapping
    except:
        pass

    # setup train log
    writer = SummaryWriter(model_dir)
    save_train_configs(config_dict, args, model_dir)
    train_log_name = os.path.join(model_dir, "train.log")
    with open(train_log_name, 'w'):
        pass

    # print training configurations
    print("Model name: %s" % model_name)
    print("Training dataset length: %d" % len(crdata_train))
    print("Batch size: %d" % config_dict['train_cfg']['batch_size'])
    print("Number of iterations in each epoch: %d" % int(len(crdata_train) / config_dict['train_cfg']['batch_size']))

    for epoch in range(epoch_start, config_dict['train_cfg']['n_epoch']):

        tic_load = time.time()
        # if epoch == epoch_start:
        #     dataloader_start = iter_start
        # else:
        #     dataloader_start = 0

        for iter, data_dict in enumerate(dataloader):
            data = data_dict['radar_data']
            image_paths = data_dict['image_paths']
            confmap_gt = data_dict['anno']['confmaps']

            if not data_dict['status']:
                # in case load npy fail
                print("Warning: Loading NPY data failed! Skip this iteration")
                tic_load = time.time()
                continue

            tic = time.time()
            optimizer.zero_grad()  # zero the parameter gradients
            confmap_preds = rodnet(data.float().cuda())

            loss_confmap = 0.
            if config_dict['model_cfg']['stacked_num'] is not None:
                for i in range(config_dict['model_cfg']['stacked_num']):
                    loss_cur = criterion(confmap_preds[i], confmap_gt.float().cuda())
                    loss_confmap += loss_cur
                loss_confmap.backward()
                optimizer.step()
            else:
                loss_confmap = criterion(confmap_preds, confmap_gt.float().cuda())
                loss_confmap.backward()
                optimizer.step()
            tic_back = time.time()

            loss_ave = np.average([loss_ave, loss_confmap.item()], weights=[iter_count, 1])

            if iter % config_dict['train_cfg']['log_step'] == 0:
                # print statistics
                load_time = tic - tic_load
                back_time = tic_back - tic
                print('epoch %2d, iter %4d: loss: %.4f (%.4f) | load time: %.2f | back time: %.2f' %
                      (epoch + 1, iter + 1, loss_confmap.item(), loss_ave, load_time, back_time))
                with open(train_log_name, 'a+') as f_log:
                    f_log.write('epoch %2d, iter %4d: loss: %.4f (%.4f) | load time: %.2f | back time: %.2f\n' %
                                (epoch + 1, iter + 1, loss_confmap.item(), loss_ave, load_time, back_time))

                writer.add_scalar('loss/loss_all', loss_confmap.item(), iter_count)
                writer.add_scalar('loss/loss_ave', loss_ave, iter_count)
                writer.add_scalar('time/time_load', load_time, iter_count)
                writer.add_scalar('time/time_back', back_time, iter_count)
                writer.add_scalar('param/param_lr', scheduler.get_last_lr()[0], iter_count)

                train_viz_path = os.path.join(model_dir, 'train_viz')
                os.makedirs(train_viz_path, exist_ok=True)
                fig_name = os.path.join(train_viz_path,
                                        '%03d_%010d_%06d.png' % (epoch + 1, iter_count + 1, iter + 1))
                vis_train(data_dict, confmap_preds, config_dict, dataset, fig_name)

            if (iter + 1) % config_dict['train_cfg']['save_step'] == 0:
                # validate current model
                # print("validing current model ...")
                # validate()

                # save current model
                save_model_path = '%s/epoch_%02d_%010d.pkl' % (model_dir, epoch + 1, iter_count + 1)
                save_model(model_name, epoch, iter, iter_count, rodnet, optimizer, loss_confmap, loss_ave,
                           save_model_path)

            iter_count += 1
            tic_load = time.time()

        # save current model
        save_model_path = '%s/epoch_%02d_final.pkl' % (model_dir, epoch + 1)
        save_model(model_name, epoch, iter, iter_count, rodnet, optimizer, loss_confmap, loss_ave,
                   save_model_path)

        scheduler.step()

    print('Training Finished.')
