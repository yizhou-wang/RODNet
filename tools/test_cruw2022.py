import os
import time
import argparse

from cruw import CRUW

from rodnet.datasets.dataset_utils import get_dataloader_test
from rodnet.datasets.CRUW2022Dataset import SPLIT_SEQ_DICT
from rodnet.models.model_utils import create_model, load_checkpoint

from rodnet.core.post_processing import post_process, post_process_single_frame
from rodnet.core.post_processing import write_dets_results, write_dets_results_single_frame
from rodnet.core.post_processing import ConfmapStack
from rodnet.core.radar_processing import chirp_amp
from rodnet.utils.visualization import visualize_test_img, visualize_test_img_wo_gt
from rodnet.utils.load_configs import load_configs_from_file, parse_cfgs, update_config_dict
from rodnet.utils.solve_dir import create_random_model_name


def parse_args():
    parser = argparse.ArgumentParser(description='Test RODNet.')

    parser.add_argument('--config', type=str, help='choose rodnet model configurations')
    parser.add_argument('--sensor_config', type=str,
                        default='./configs/dataset_configs/sensor_config_cruw2022.json')
    # parser.add_argument('--data_dir', type=str, default='./data/', help='directory to the prepared data')
    parser.add_argument('--checkpoint', type=str, help='path to the saved trained model')
    parser.add_argument('--res_dir', type=str, default='./results/', help='directory to save testing results')
    parser.add_argument('--demo', action="store_true", help='False: test with GT, True: demo without GT')
    parser.add_argument('--symbol', action="store_true", help='use symbol or text+score')

    parser = parse_cfgs(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    sybl = args.symbol
    config_dict = load_configs_from_file(args.config)
    config_dict = update_config_dict(config_dict, args)  # update configs by args
    dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], sensor_config_name=args.sensor_config)

    radar_configs = dataset.sensor_cfg.radar_cfg
    win_size = config_dict['train_cfg']['win_size']
    n_class = dataset.object_cfg.n_class
    confmap_shape = (n_class, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        cp_path = args.checkpoint
    else:
        raise ValueError("No trained model found.")

    rodnet = create_model(config_dict, args, dataset, is_train=False)
    print('loading pretrained model %s ...' % cp_path)
    rodnet = load_checkpoint(rodnet, cp_path, is_train=False)
    rodnet.eval()

    model_name = create_random_model_name(config_dict['model_cfg']['name'], cp_path)
    test_res_dir = os.path.join(os.path.join(args.res_dir, model_name))
    os.makedirs(test_res_dir, exist_ok=True)

    # save current checkpoint path
    weight_log_path = os.path.join(test_res_dir, 'weight_name.txt')
    if os.path.exists(weight_log_path):
        with open(weight_log_path, 'a+') as f:
            f.write(cp_path + '\n')
    else:
        with open(weight_log_path, 'w') as f:
            f.write(cp_path + '\n')

    if args.demo:
        seq_names = SPLIT_SEQ_DICT['demo']
    else:
        seq_names = SPLIT_SEQ_DICT['test']
    print('sequences to be tested:', seq_names)

    total_time = 0
    total_count = 0

    for seq_name in seq_names:
        seq_res_dir = os.path.join(test_res_dir, seq_name)
        print('saving results to %s ...' % seq_res_dir)
        os.makedirs(seq_res_dir, exist_ok=True)
        seq_res_viz_dir = os.path.join(seq_res_dir, 'rod_viz')
        os.makedirs(seq_res_viz_dir, exist_ok=True)
        f = open(os.path.join(seq_res_dir, 'rod_res.txt'), 'w')
        f.close()

        print('testing sequence %s ...' % seq_name)
        # setup dataloader
        crdata_test, dataloader = get_dataloader_test(dataset.dataset, config_dict, args, dataset)

        init_genConfmap = ConfmapStack(confmap_shape)
        iter_ = init_genConfmap
        for i in range(win_size - 1):
            while iter_.next is not None:
                iter_ = iter_.next
            iter_.next = ConfmapStack(confmap_shape)

        load_tic = time.time()
        for start_frame_id, data_dict in enumerate(dataloader):
            if start_frame_id % config_dict['test_cfg']['test_stride'] != 0:
                continue
            load_time = time.time() - load_tic
            data = data_dict['radar_data']
            try:
                image_paths = data_dict['image_paths'][0]
            except:
                print('warning: fail to load RGB images, will not visualize results')
                image_paths = None
            # seq_name = data_dict['seq_names'][0]
            if not args.demo:
                confmap_gt = data_dict['anno']['confmaps']
                obj_info = data_dict['anno']['obj_infos']
            else:
                confmap_gt = None
                obj_info = None

            save_path = os.path.join(test_res_dir, seq_name, 'rod_res.txt')

            tic = time.time()
            confmap_pred = rodnet(data.float().cuda())
            if config_dict['model_cfg']['stacked_num'] is not None:
                confmap_pred = confmap_pred[-1].cpu().detach().numpy()  # (1, 4, 32, 128, 128)
            else:
                confmap_pred = confmap_pred.cpu().detach().numpy()

            if args.use_noise_channel:
                confmap_pred = confmap_pred[:, :n_class, :, :, :]

            infer_time = time.time() - tic
            total_time += infer_time

            iter_ = init_genConfmap
            for i in range(confmap_pred.shape[2]):
                if iter_.next is None and i != confmap_pred.shape[2] - 1:
                    iter_.next = ConfmapStack(confmap_shape)
                iter_.append(confmap_pred[0, :, i, :, :])
                iter_ = iter_.next

            process_tic = time.time()
            for i in range(config_dict['test_cfg']['test_stride']):
                total_count += 1
                res_final = post_process_single_frame(init_genConfmap.confmap, dataset, config_dict)
                cur_frame_id = start_frame_id + i
                write_dets_results_single_frame(res_final, cur_frame_id, save_path, dataset)
                confmap_pred_0 = init_genConfmap.confmap
                res_final_0 = res_final
                if image_paths is not None:
                    img_path = image_paths[i]
                    radar_input = chirp_amp(data.numpy()[0, :, i, 0, :, :], radar_configs['data_type'])
                    fig_name = os.path.join(test_res_dir, seq_name, 'rod_viz', '%010d.jpg' % (cur_frame_id))
                    if confmap_gt is not None:
                        confmap_gt_0 = confmap_gt[0, :, i, :, :]
                        visualize_test_img(fig_name, img_path, radar_input, confmap_pred_0, confmap_gt_0, res_final_0,
                                           dataset, sybl=sybl)
                    else:
                        visualize_test_img_wo_gt(fig_name, img_path, radar_input, confmap_pred_0, res_final_0,
                                                 dataset, sybl=sybl)
                init_genConfmap = init_genConfmap.next

            if iter == len(dataloader) - 1:
                offset = config_dict['test_cfg']['test_stride']
                cur_frame_id = start_frame_id + offset
                while init_genConfmap is not None:
                    total_count += 1
                    res_final = post_process_single_frame(init_genConfmap.confmap, dataset, config_dict)
                    write_dets_results_single_frame(res_final, cur_frame_id, save_path, dataset)
                    confmap_pred_0 = init_genConfmap.confmap
                    res_final_0 = res_final
                    if image_paths is not None:
                        img_path = image_paths[offset]
                        radar_input = chirp_amp(data.numpy()[0, :, offset, :, :], radar_configs['data_type'])
                        fig_name = os.path.join(test_res_dir, seq_name, 'rod_viz', '%010d.jpg' % (cur_frame_id))
                        if confmap_gt is not None:
                            confmap_gt_0 = confmap_gt[0, :, offset, :, :]
                            visualize_test_img(fig_name, img_path, radar_input, confmap_pred_0, confmap_gt_0,
                                               res_final_0,
                                               dataset, sybl=sybl)
                        else:
                            visualize_test_img_wo_gt(fig_name, img_path, radar_input, confmap_pred_0, res_final_0,
                                                     dataset, sybl=sybl)
                    init_genConfmap = init_genConfmap.next
                    offset += 1
                    cur_frame_id += 1

            if init_genConfmap is None:
                init_genConfmap = ConfmapStack(confmap_shape)

            proc_time = time.time() - process_tic
            print("Testing %s: frame %4d to %4d | Load time: %.4f | Inference time: %.4f | Process time: %.4f" %
                  (seq_name, start_frame_id, start_frame_id + win_size, load_time, infer_time, proc_time))

            load_tic = time.time()

    print("ave time: %f" % (total_time / total_count))
