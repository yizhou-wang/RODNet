import math

from cruw.mapping.coor_transform import pol2cart_ramap

from rodnet.core import get_class_name


def get_ols_btw_objects(obj1, obj2, dataset):
    classes = dataset.object_cfg.classes
    object_sizes = dataset.object_cfg.sizes

    if obj1['class_id'] != obj2['class_id']:
        print('Error: Computing OLS between different classes!')
        raise TypeError("OLS can only be compute between objects with same class.  ")
    if obj1['score'] < obj2['score']:
        raise TypeError("Confidence score of obj1 should not be smaller than obj2. "
                        "obj1['score'] = %s, obj2['score'] = %s" % (obj1['score'], obj2['score']))

    classid = obj1['class_id']
    class_str = get_class_name(classid, classes)
    rng1 = obj1['range']
    agl1 = obj1['angle']
    rng2 = obj2['range']
    agl2 = obj2['angle']
    x1, y1 = pol2cart_ramap(rng1, agl1)
    x2, y2 = pol2cart_ramap(rng2, agl2)
    dx = x1 - x2
    dy = y1 - y2
    s_square = x1 ** 2 + y1 ** 2
    kappa = object_sizes[class_str] / 100  # TODO: tune kappa
    e = (dx ** 2 + dy ** 2) / 2 / (s_square * kappa)
    ols = math.exp(-e)
    return ols


def get_ols_btw_pts(pt1, pt2, class_id, dataset):
    classes = dataset.object_cfg.classes
    object_sizes = dataset.object_cfg.sizes

    class_str = get_class_name(class_id, classes)
    x1, y1 = pol2cart_ramap(pt1[0], pt1[1])
    x2, y2 = pol2cart_ramap(pt2[0], pt2[1])
    dx = x1 - x2
    dy = y1 - y2
    s_square = x1 ** 2 + y1 ** 2
    kappa = object_sizes[class_str] / 100  # TODO: tune kappa
    e = (dx ** 2 + dy ** 2) / 2 / (s_square * kappa)
    ols = math.exp(-e)
    return ols


def ols(dist, s, kappa):
    e = dist ** 2 / 2 / (s ** 2 * kappa)
    return math.exp(-e)
