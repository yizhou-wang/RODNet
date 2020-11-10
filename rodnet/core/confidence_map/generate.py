import numpy as np
import math

from rodnet.core.object_class import get_class_id


def generate_confmap(n_obj, obj_info, dataset, config_dict, gaussian_thres=36):
    """
    Generate confidence map a radar frame.
    :param n_obj: number of objects in this frame
    :param obj_info: obj_info includes metadata information
    :param dataset: dataset object
    :param config_dict: rodnet configurations
    :param gaussian_thres: threshold for gaussian distribution in confmaps
    :return: generated confmap
    """
    n_class = dataset.object_cfg.n_class
    classes = dataset.object_cfg.classes
    radar_configs = dataset.sensor_cfg.radar_cfg
    confmap_sigmas = config_dict['confmap_cfg']['confmap_sigmas']
    confmap_sigmas_interval = config_dict['confmap_cfg']['confmap_sigmas_interval']
    confmap_length = config_dict['confmap_cfg']['confmap_length']

    range_grid = dataset.range_grid
    angle_grid = dataset.angle_grid

    confmap = np.zeros((n_class, radar_configs['ramap_rsize'], radar_configs['ramap_asize']), dtype=float)
    for objid in range(n_obj):
        rng_idx = obj_info['center_ids'][objid][0]
        agl_idx = obj_info['center_ids'][objid][1]
        class_name = obj_info['categories'][objid]
        if class_name not in classes:
            # print("not recognized class: %s" % class_name)
            continue
        class_id = get_class_id(class_name, classes)
        sigma = 2 * np.arctan(confmap_length[class_name] / (2 * range_grid[rng_idx])) * confmap_sigmas[class_name]
        sigma_interval = confmap_sigmas_interval[class_name]
        if sigma > sigma_interval[1]:
            sigma = sigma_interval[1]
        if sigma < sigma_interval[0]:
            sigma = sigma_interval[0]
        for i in range(radar_configs['ramap_rsize']):
            for j in range(radar_configs['ramap_asize']):
                distant = (((rng_idx - i) * 2) ** 2 + (agl_idx - j) ** 2) / sigma ** 2
                if distant < gaussian_thres:  # threshold for confidence maps
                    value = np.exp(- distant / 2) / (2 * math.pi)
                    confmap[class_id, i, j] = value if value > confmap[class_id, i, j] else confmap[class_id, i, j]

    return confmap
