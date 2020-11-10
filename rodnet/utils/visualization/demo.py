import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from rodnet.core.object_class import get_class_name
from .fig_configs import fig, fp, symbols


def visualize_train_img_old(fig_name, input_radar, output_confmap, confmap_gt):
    fig = plt.figure(figsize=(8, 4))
    img = input_radar
    fig.add_subplot(1, 3, 1)
    plt.imshow(img, vmin=0, vmax=1, origin='lower', aspect='auto')
    img = output_confmap
    fig.add_subplot(1, 3, 2)
    plt.imshow(img, vmin=0, vmax=1, origin='lower', aspect='auto')
    img = confmap_gt
    fig.add_subplot(1, 3, 3)
    plt.imshow(img, vmin=0, vmax=1, origin='lower', aspect='auto')
    plt.savefig(fig_name)
    plt.close(fig)


def visualize_train_img(fig_name, img_path, input_radar, output_confmap, confmap_gt):
    fig = plt.figure(figsize=(8, 8))
    img_data = mpimg.imread(img_path)

    fig.add_subplot(2, 2, 1)
    plt.imshow(img_data.astype(np.uint8))

    fig.add_subplot(2, 2, 2)
    plt.imshow(input_radar, origin='lower', aspect='auto')

    fig.add_subplot(2, 2, 3)
    output_confmap = np.transpose(output_confmap, (1, 2, 0))
    output_confmap[output_confmap < 0] = 0
    plt.imshow(output_confmap, vmin=0, vmax=1, origin='lower', aspect='auto')

    fig.add_subplot(2, 2, 4)
    confmap_gt = np.transpose(confmap_gt, (1, 2, 0))
    plt.imshow(confmap_gt, vmin=0, vmax=1, origin='lower', aspect='auto')

    plt.savefig(fig_name)
    plt.close(fig)


def visualize_test_img(fig_name, img_path, input_radar, confmap_pred, confmap_gt, res_final, dataset, viz=False,
                       sybl=False):
    max_dets, _ = res_final.shape
    classes = dataset.object_cfg.classes

    img_data = mpimg.imread(img_path)
    if img_data.shape[0] > 864:
        img_data = img_data[:img_data.shape[0] // 5 * 4, :, :]

    fig.add_subplot(2, 2, 1)
    plt.imshow(img_data.astype(np.uint8))
    plt.axis('off')
    plt.title("Image")

    fig.add_subplot(2, 2, 2)
    plt.imshow(input_radar, origin='lower', aspect='auto')
    plt.axis('off')
    plt.title("RA Heatmap")

    fig.add_subplot(2, 2, 3)
    confmap_pred = np.transpose(confmap_pred, (1, 2, 0))
    confmap_pred[confmap_pred < 0] = 0
    confmap_pred[confmap_pred > 1] = 1
    plt.imshow(confmap_pred, vmin=0, vmax=1, origin='lower', aspect='auto')
    for d in range(max_dets):
        cla_id = int(res_final[d, 0])
        if cla_id == -1:
            continue
        row_id = res_final[d, 1]
        col_id = res_final[d, 2]
        conf = res_final[d, 3]
        conf = 1.0 if conf > 1 else conf
        cla_str = get_class_name(cla_id, classes)
        if sybl:
            text = symbols[cla_str]
            plt.text(col_id, row_id + 3, text, fontproperties=fp, color='white', size=20, ha="center")
        else:
            plt.scatter(col_id, row_id, s=10, c='white')
            text = cla_str + '\n%.2f' % conf
            plt.text(col_id + 5, row_id, text, color='white', fontsize=10)
    plt.axis('off')
    plt.title("RODNet Detection")

    fig.add_subplot(2, 2, 4)
    confmap_gt = np.transpose(confmap_gt, (1, 2, 0))
    plt.imshow(confmap_gt, vmin=0, vmax=1, origin='lower', aspect='auto')
    plt.axis('off')
    plt.title("Ground Truth")

    plt.savefig(fig_name)
    if viz:
        plt.pause(0.1)
    plt.clf()


def visualize_test_img_wo_gt(fig_name, img_path, input_radar, confmap_pred, res_final, dataset, viz=False,
                             sybl=False):
    max_dets, _ = res_final.shape
    classes = dataset.object_cfg.classes

    fig.set_size_inches(12, 4)
    img_data = mpimg.imread(img_path)
    if img_data.shape[0] > 864:
        img_data = img_data[:img_data.shape[0] // 5 * 4, :, :]

    fig.add_subplot(1, 3, 1)
    plt.imshow(img_data.astype(np.uint8))
    plt.axis('off')
    plt.title("RGB Image")

    fig.add_subplot(1, 3, 2)
    input_radar[input_radar < 0] = 0
    input_radar[input_radar > 1] = 1
    plt.imshow(input_radar, vmin=0, vmax=1, origin='lower', aspect='auto')
    plt.axis('off')
    plt.title("RF Image")

    fig.add_subplot(1, 3, 3)
    confmap_pred = np.transpose(confmap_pred, (1, 2, 0))
    confmap_pred[confmap_pred < 0] = 0
    confmap_pred[confmap_pred > 1] = 1
    plt.imshow(confmap_pred, vmin=0, vmax=1, origin='lower', aspect='auto')
    for d in range(max_dets):
        cla_id = int(res_final[d, 0])
        if cla_id == -1:
            continue
        row_id = res_final[d, 1]
        col_id = res_final[d, 2]
        conf = res_final[d, 3]
        conf = 1.0 if conf > 1 else conf
        cla_str = get_class_name(cla_id, classes)
        if sybl:
            text = symbols[cla_str]
            plt.text(col_id - 3, row_id + 2, text, fontproperties=fp, color='white', size=20)
        else:
            plt.scatter(col_id, row_id, s=10, c='white')
            text = cla_str + '\n%.2f' % conf
            plt.text(col_id + 5, row_id, text, color='white', fontsize=10)
    plt.axis('off')
    plt.title("RODNet Detections")

    plt.savefig(fig_name)
    if viz:
        plt.pause(0.1)
    plt.clf()
