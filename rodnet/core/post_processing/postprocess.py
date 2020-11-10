import numpy as np

from .ops import detect_peaks
from .lnms import lnms
from rodnet.utils.visualization import visualize_postprocessing


def search_surround(peak_conf, row, col, conf_valu, search_size):
    height = peak_conf.shape[0]
    width = peak_conf.shape[1]
    half_size = int((search_size - 1) / 2)
    row_start = max(half_size, row - half_size)
    row_end = min(height - half_size - 1, row + half_size)
    col_start = max(half_size, col - half_size)
    col_end = min(width - half_size - 1, col + half_size)
    # print(row_start)
    No_bigger = True
    for i in range(row_start, row_end + 1):
        for j in range(col_start, col_end + 1):
            if peak_conf[i, j] > conf_valu:
                # current conf is not big enough, skip this peak
                No_bigger = False
                break

    return No_bigger, [row_start, row_end, col_start, col_end]


def peak_mapping(peak_conf, peak_class, list_row, list_col, confmap, search_size, o_class):
    for i in range(len(list_col)):
        row_id = list_row[i]
        col_id = list_col[i]
        conf_valu = confmap[row_id, col_id]

        flag, indices = search_surround(peak_conf, row_id, col_id, conf_valu, search_size)
        if flag:
            # clear all detections in search window
            search_width = indices[1] - indices[0] + 1
            search_height = indices[3] - indices[2] + 1
            peak_conf[indices[0]:indices[1] + 1, indices[2]:indices[3] + 1] = np.zeros((search_width, search_height))
            peak_class[indices[0]:indices[1] + 1, indices[2]:indices[3] + 1] = - np.ones((search_width, search_height))
            # write the detected objects to matrix
            peak_conf[row_id, col_id] = conf_valu
            peak_class[row_id, col_id] = class_ids[o_class]

    return peak_conf, peak_class


def find_greatest_points(peak_conf, peak_class):
    detect_mat = - np.ones((rodnet_configs['max_dets'], 4))
    height = peak_conf.shape[0]
    width = peak_conf.shape[1]
    peak_flatten = peak_conf.flatten()
    indic = np.argsort(peak_flatten)
    ind_len = indic.shape[0]

    if ind_len >= rodnet_configs['max_dets']:
        choos_ind = np.flip(indic[-rodnet_configs['max_dets']:ind_len])
    else:
        choos_ind = np.flip(indic)

    for count, ele_ind in enumerate(choos_ind):
        row = ele_ind // width
        col = ele_ind % width
        if peak_conf[row, col] > 0:
            detect_mat[count, 0] = peak_class[row, col]
            detect_mat[count, 1] = row
            detect_mat[count, 2] = col
            detect_mat[count, 3] = peak_conf[row, col]

    return detect_mat


def post_process(confmaps, config_dict):
    """
    Post-processing for RODNet
    :param confmaps: predicted confidence map [B, n_class, win_size, ramap_r, ramap_a]
    :param search_size: search other detections within this window (resolution of our system)
    :param peak_thres: peak threshold
    :return: [B, win_size, max_dets, 4]
    """
    n_class = config_dict['class_cfg']['n_class']
    model_configs = config_dict['model_cfg']
    rng_grid = config_dict['mappings']['range_grid']
    agl_grid = config_dict['mappings']['angle_grid']
    max_dets = model_configs['max_dets']
    peak_thres = model_configs['peak_thres']

    batch_size, class_size, win_size, height, width = confmaps.shape

    if class_size != n_class:
        raise TypeError("Wrong class number setting. ")

    res_final = - np.ones((batch_size, win_size, max_dets, 4))

    for b in range(batch_size):
        for w in range(win_size):
            detect_mat = []
            for c in range(class_size):
                obj_dicts_in_class = []
                confmap = np.squeeze(confmaps[b, c, w, :, :])
                rowids, colids = detect_peaks(confmap, threshold=peak_thres)

                for ridx, aidx in zip(rowids, colids):
                    rng = rng_grid[ridx]
                    agl = agl_grid[aidx]
                    conf = confmap[ridx, aidx]
                    obj_dict = {'frameid': None, 'range': rng, 'angle': agl, 'ridx': ridx, 'aidx': aidx,
                                'classid': c, 'score': conf}
                    obj_dicts_in_class.append(obj_dict)

                detect_mat_in_class = lnms(obj_dicts_in_class, config_dict)
                detect_mat.append(detect_mat_in_class)

            detect_mat = np.array(detect_mat)
            detect_mat = np.reshape(detect_mat, (class_size * max_dets, 4))
            detect_mat = detect_mat[detect_mat[:, 3].argsort(kind='mergesort')[::-1]]
            res_final[b, w, :, :] = detect_mat[:max_dets]

    return res_final


def post_process_single_frame(confmaps, dataset, config_dict):
    """
    Post-processing for RODNet
    :param confmaps: predicted confidence map [B, n_class, win_size, ramap_r, ramap_a]
    :param search_size: search other detections within this window (resolution of our system)
    :param peak_thres: peak threshold
    :return: [B, win_size, max_dets, 4]
    """
    n_class = dataset.object_cfg.n_class
    rng_grid = dataset.range_grid
    agl_grid = dataset.angle_grid
    model_configs = config_dict['model_cfg']
    max_dets = model_configs['max_dets']
    peak_thres = model_configs['peak_thres']

    class_size, height, width = confmaps.shape

    if class_size != n_class:
        raise TypeError("Wrong class number setting. ")

    res_final = - np.ones((max_dets, 4))

    detect_mat = []
    for c in range(class_size):
        obj_dicts_in_class = []
        confmap = confmaps[c, :, :]
        rowids, colids = detect_peaks(confmap, threshold=peak_thres)

        for ridx, aidx in zip(rowids, colids):
            rng = rng_grid[ridx]
            agl = agl_grid[aidx]
            conf = confmap[ridx, aidx]
            obj_dict = dict(
                frame_id=None,
                range=rng,
                angle=agl,
                range_id=ridx,
                angle_id=aidx,
                class_id=c,
                score=conf,
            )
            obj_dicts_in_class.append(obj_dict)

        detect_mat_in_class = lnms(obj_dicts_in_class, dataset, config_dict)
        detect_mat.append(detect_mat_in_class)

    detect_mat = np.array(detect_mat)
    detect_mat = np.reshape(detect_mat, (class_size * max_dets, 4))
    detect_mat = detect_mat[detect_mat[:, 3].argsort(kind='mergesort')[::-1]]
    res_final[:, :] = detect_mat[:max_dets]

    return res_final


if __name__ == "__main__":
    input_test = np.random.random_sample((1, 3, 16, 122, 91))
    res_final = post_process(input_test)
    for b in range(1):
        for w in range(16):
            confmaps = np.squeeze(input_test[b, :, w, :, :])
            visualize_postprocessing(confmaps, res_final[b, w, :, :])
