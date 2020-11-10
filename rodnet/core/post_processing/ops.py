import numpy as np


def detect_peaks(image, threshold=0.3):
    peaks_row = []
    peaks_col = []
    height, width = image.shape
    for h in range(1, height - 1):
        for w in range(2, width - 2):
            area = image[h - 1:h + 2, w - 2:w + 3]
            center = image[h, w]
            flag = np.where(area >= center)
            if flag[0].shape[0] == 1 and center > threshold:
                peaks_row.append(h)
                peaks_col.append(w)

    return peaks_row, peaks_col
