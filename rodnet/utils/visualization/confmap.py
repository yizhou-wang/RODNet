import numpy as np
import matplotlib.pyplot as plt


def visualize_confmap(confmap, pps=[]):
    if len(confmap.shape) == 2:
        plt.imshow(confmap, origin='lower', aspect='auto')
        for pp in pps:
            plt.scatter(pp[1], pp[0], s=5, c='white')
        plt.show()
        return
    else:
        n_channel, _, _ = confmap.shape
    if n_channel == 3:
        confmap_viz = np.transpose(confmap, (1, 2, 0))
    elif n_channel > 3:
        confmap_viz = np.transpose(confmap[:3, :, :], (1, 2, 0))
        if n_channel == 4:
            confmap_noise = confmap[3, :, :]
            plt.imshow(confmap_noise, origin='lower', aspect='auto')
            plt.show()
    else:
        print("Warning: wrong shape of confmap!")
        return
    plt.imshow(confmap_viz, origin='lower', aspect='auto')
    for pp in pps:
        plt.scatter(pp[1], pp[0], s=5, c='white')
    plt.show()


def visualize_confmaps_cr(confmapc, confmapr, confmapcr, ppsc=[], ppsr=[], ppres=[], figname=None):
    fig = plt.figure(figsize=(8, 8))
    n_channel, nr, na = confmapc.shape
    fig_id = 1
    for class_id in range(n_channel):
        fig.add_subplot(n_channel, 3, fig_id)
        fig_id += 1
        plt.imshow(confmapc[class_id], origin='lower', aspect='auto')
        for pp in ppsc[class_id]:
            plt.scatter(pp[1], pp[0], s=5, c='white')
        plt.xlim(0, na)
        plt.ylim(0, nr)

        fig.add_subplot(n_channel, 3, fig_id)
        fig_id += 1
        plt.imshow(confmapr, origin='lower', aspect='auto')
        for pp in ppsr:
            plt.scatter(pp[1], pp[0], s=5, c='white')
        plt.xlim(0, na)
        plt.ylim(0, nr)

        fig.add_subplot(n_channel, 3, fig_id)
        fig_id += 1
        plt.imshow(confmapcr[class_id], origin='lower', aspect='auto', vmin=0, vmax=1)
        for pp in ppres[class_id]:
            plt.scatter(pp[1], pp[0], s=5, c='white')
        plt.xlim(0, na)
        plt.ylim(0, nr)

    if figname is None:
        plt.show()
    else:
        plt.savefig(figname)
        plt.close(fig)


def visualize_postprocessing(confmaps, det_results):
    confmap_pred = np.transpose(confmaps, (1, 2, 0))
    plt.imshow(confmap_pred, vmin=0, vmax=1, origin='lower', aspect='auto')
    for d in range(rodnet_configs['max_dets']):
        cla_id = int(det_results[d, 0])
        if cla_id == -1:
            continue
        row_id = det_results[d, 1]
        col_id = det_results[d, 2]
        conf = det_results[d, 3]
        cla_str = class_table[cla_id]
        plt.scatter(col_id, row_id, s=50, c='white')
        plt.text(col_id + 5, row_id, cla_str + '\n%.2f' % conf, color='white', fontsize=10, fontweight='black')
    plt.axis('off')
    plt.title("RODNet Detection")
    plt.show()
