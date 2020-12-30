import numpy as np


def chirp_amp(chirp, radar_data_type):
    """
    Calculate amplitude of a chirp
    :param chirp: radar data of one chirp (w x h x 2) or (2 x w x h)
    :param radar_data_type: current available types include 'RI', 'RISEP', 'AP', 'APSEP'
    :return: amplitude map for the input chirp (w x h)
    """
    c0, c1, c2 = chirp.shape
    if radar_data_type == 'RI' or radar_data_type == 'RISEP' or radar_data_type == 'ROD2021':
        if c0 == 2:
            chirp_abs = np.sqrt(chirp[0, :, :] ** 2 + chirp[1, :, :] ** 2)
        elif c2 == 2:
            chirp_abs = np.sqrt(chirp[:, :, 0] ** 2 + chirp[:, :, 1] ** 2)
        else:
            raise ValueError
    elif radar_data_type == 'AP' or radar_data_type == 'APSEP':
        if c0 == 2:
            chirp_abs = chirp[0, :, :]
        elif c2 == 2:
            chirp_abs = chirp[:, :, 0]
        else:
            raise ValueError
    else:
        raise ValueError
    return chirp_abs
