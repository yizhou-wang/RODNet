import numpy as np
import math


def normalize_confmap(confmap):
    conf_min = np.min(confmap)
    conf_max = np.max(confmap)
    if conf_max - conf_min != 0:
        confmap_norm = (confmap - conf_min) / (conf_max - conf_min)
    else:
        confmap_norm = confmap
    return confmap_norm


def add_noise_channel(confmap, dataset, config_dict):
    n_class = dataset.object_cfg.n_class
    radar_configs = dataset.sensor_cfg.radar_cfg

    confmap_new = np.zeros((n_class + 1, radar_configs['ramap_rsize'], radar_configs['ramap_asize']), dtype=float)
    confmap_new[:n_class, :, :] = confmap
    conf_max = np.max(confmap, axis=0)
    confmap_new[n_class, :, :] = 1.0 - conf_max
    return confmap_new


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def dist_point_segment(point, segment):
    x3, y3 = point
    (x1, y1), (x2, y2) = segment

    px = x2 - x1
    py = y2 - y1
    norm = px * px + py * py
    u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = (dx * dx + dy * dy) ** .5

    return dist, (x, y)


def rotate_conf_pattern(dx, dy, ori):
    dr = (dx * dx + dy * dy) ** 0.5
    dtheta = math.atan2(dy, dx)
    dtheta -= ori
    dx_new = dr * math.cos(dtheta)
    dy_new = dr * math.sin(dtheta)
    return dx_new, dy_new


# A utility function to calculate area
# of triangle formed by (x1, y1),
# (x2, y2) and (x3, y3)
def area(x1, y1, x2, y2, x3, y3):
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)


# A function to check whether point P(x, y)
# lies inside the triangle formed by
# A(x1, y1), B(x2, y2) and C(x3, y3)
def is_inside_triangle(p1, p2, p3, p):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x, y = p

    # Calculate area of triangle ABC
    A = area(x1, y1, x2, y2, x3, y3)
    # Calculate area of triangle PBC
    A1 = area(x, y, x2, y2, x3, y3)
    # Calculate area of triangle PAC
    A2 = area(x1, y1, x, y, x3, y3)
    # Calculate area of triangle PAB
    A3 = area(x1, y1, x2, y2, x, y)

    # Check if sum of A1, A2 and A3
    # is same as A
    if A == A1 + A2 + A3:
        return True
    else:
        return False
