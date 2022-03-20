from rodnet.core.radar_processing import chirp_amp
from rodnet.utils.visualization import visualize_train_img


def vis_train(data_dict, confmap_preds, config_dict, dataset, fig_name):
    stacked_num = config_dict['model_cfg']['stacked_num']
    n_class = dataset.object_cfg.n_class
    data = data_dict['radar_data']
    image_paths = data_dict['image_paths']
    confmap_gt = data_dict['anno']['confmaps']

    if stacked_num is not None:
        confmap_pred = confmap_preds[stacked_num - 1].cpu().detach().numpy()
    else:
        confmap_pred = confmap_preds.cpu().detach().numpy()

    if 'mnet_cfg' in config_dict['model_cfg']:
        chirp_amp_curr = chirp_amp(data.numpy()[0, :, 0, 0, :, :], dataset.sensor_cfg.radar_cfg['data_type'])
    else:
        chirp_amp_curr = chirp_amp(data.numpy()[0, :, 0, :, :], dataset.sensor_cfg.radar_cfg['data_type'])

    # draw train images
    img_path = image_paths[0][0]
    visualize_train_img(fig_name, img_path, chirp_amp_curr,
                        confmap_pred[0, :n_class, 0, :, :],
                        confmap_gt[0, :n_class, 0, :, :])
