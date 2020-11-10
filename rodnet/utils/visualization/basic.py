import numpy as np
import matplotlib.pyplot as plt


def heatmap2rgb(heatmap):
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(heatmap)
    rgb_img = np.delete(rgba_img, 3, 2)
    rgb_img = np.transpose(rgb_img, (2, 0, 1))
    return rgb_img
