import numpy as np
import matplotlib.pyplot as plt

from rodnet.core.radar_processing.chirp_ops import chirp_amp


def visualize_radar_chirp(chirp, radar_data_type):
    """
    Visualize radar data of one chirp
    :param chirp: (w x h x 2) or (2 x w x h)
    :param radar_data_type: current available types include 'RI', 'RISEP', 'AP', 'APSEP'
    :return:
    """
    chirp_abs = chirp_amp(chirp, radar_data_type)
    plt.imshow(chirp_abs)
    plt.show()


def visualize_radar_chirps(chirps, radar_data_type):
    """
    Visualize radar data of multiple chirps
    :param chirps: (N x w x h x 2) or (N x 2 x w x h)
    :param radar_data_type: current available types include 'RI', 'RISEP', 'AP', 'APSEP'
    :return:
    """
    num_chirps, c0, c1, c2 = chirps.shape
    if c2 == 2:
        chirps_abs = np.zeros((num_chirps, c0, c1))
    elif c0 == 2:
        chirps_abs = np.zeros((num_chirps, c1, c2))
    else:
        raise ValueError
    for chirp_id in range(num_chirps):
        chirps_abs[chirp_id, :, :] = chirp_amp(chirps[chirp_id, :, :, :], radar_data_type)
    chirp_abs_avg = np.mean(chirps_abs, axis=0)
    plt.imshow(chirp_abs_avg)
    plt.show()


def visualize_fuse_crdets(chirp, obj_dicts, figname=None, viz=False):
    chirp_abs = chirp_amp(chirp)
    chirp_shape = chirp_abs.shape
    plt.figure()
    plt.imshow(chirp_abs, vmin=0, vmax=1, origin='lower')

    for obj_id, obj_dict in enumerate(obj_dicts):
        plt.scatter(obj_dict['angle_id'], obj_dict['range_id'], s=10, c='white')
        try:
            text = str(obj_dict['object_id']) + ' ' + obj_dict['class']
        except:
            text = str(obj_dict['object_id'])
        plt.text(obj_dict['angle_id'] + 5, obj_dict['range_id'], text, color='white', fontsize=10)

    plt.xlim(0, chirp_shape[1])
    plt.ylim(0, chirp_shape[0])

    if viz:
        plt.show()
    else:
        plt.savefig(figname)
        plt.close()


def visualize_fuse_crdets_compare(img_path, chirp, c_dicts, r_dicts, cr_dicts, figname=None, viz=False):
    chirp_abs = chirp_amp(chirp)
    chirp_shape = chirp_abs.shape
    fig_local = plt.figure()
    fig_local.set_size_inches(16, 4)

    fig_local.add_subplot(1, 4, 1)
    im = plt.imread(img_path)
    plt.imshow(im)

    fig_local.add_subplot(1, 4, 2)
    plt.imshow(chirp_abs, vmin=0, vmax=1, origin='lower')
    for obj_id, obj_dict in enumerate(c_dicts):
        plt.scatter(obj_dict['angle_id'], obj_dict['range_id'], s=10, c='white')
        try:
            obj_dict['object_id']
        except:
            obj_dict['object_id'] = ''
        try:
            text = str(obj_dict['object_id']) + ' ' + obj_dict['class']
        except:
            text = str(obj_dict['object_id'])
        plt.text(obj_dict['angle_id'] + 5, obj_dict['range_id'], text, color='white', fontsize=10)
    plt.xlim(0, chirp_shape[1])
    plt.ylim(0, chirp_shape[0])

    fig_local.add_subplot(1, 4, 3)
    plt.imshow(chirp_abs, vmin=0, vmax=1, origin='lower')
    for obj_id, obj_dict in enumerate(r_dicts):
        plt.scatter(obj_dict['angle_id'], obj_dict['range_id'], s=10, c='white')
        try:
            obj_dict['object_id']
        except:
            obj_dict['object_id'] = ''
        try:
            text = str(obj_dict['object_id']) + ' ' + obj_dict['class']
        except:
            text = str(obj_dict['object_id'])
        plt.text(obj_dict['angle_id'] + 5, obj_dict['range_id'], text, color='white', fontsize=10)
    plt.xlim(0, chirp_shape[1])
    plt.ylim(0, chirp_shape[0])

    fig_local.add_subplot(1, 4, 4)
    plt.imshow(chirp_abs, vmin=0, vmax=1, origin='lower')
    for obj_id, obj_dict in enumerate(cr_dicts):
        plt.scatter(obj_dict['angle_id'], obj_dict['range_id'], s=10, c='white')
        try:
            obj_dict['object_id']
        except:
            obj_dict['object_id'] = '%.2f' % obj_dict['confidence']
        try:
            text = str(obj_dict['object_id']) + ' ' + obj_dict['class']
        except:
            text = str(obj_dict['object_id'])
        plt.text(obj_dict['angle_id'] + 5, obj_dict['range_id'], text, color='white', fontsize=10)
    plt.xlim(0, chirp_shape[1])
    plt.ylim(0, chirp_shape[0])

    if viz:
        plt.show()
    else:
        plt.savefig(figname)
        plt.close()


def visualize_anno_ramap(chirp, obj_info, figname, viz=False):
    chirp_abs = chirp_amp(chirp)
    plt.figure()
    plt.imshow(chirp_abs, vmin=0, vmax=1, origin='lower')

    for obj in obj_info:
        rng_idx, agl_idx, class_id = obj
        if class_id >= 0:
            try:
                cla_str = class_table[class_id]
            except:
                continue
        else:
            continue
        plt.scatter(agl_idx, rng_idx, s=10, c='white')
        plt.text(agl_idx + 5, rng_idx, cla_str, color='white', fontsize=10)

    if viz:
        plt.show()
    else:
        plt.savefig(figname)
        plt.close()
