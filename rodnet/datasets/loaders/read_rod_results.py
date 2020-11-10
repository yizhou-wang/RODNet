import math

from rodnet.core.object_class import get_class_id


def load_rodnet_res(filename, config_dict):
    n_class = config_dict['class_cfg']['n_class']
    classes = config_dict['class_cfg']['classes']
    model_configs = config_dict['model_cfg']
    rng_grid = config_dict['mappings']['range_grid']
    agl_grid = config_dict['mappings']['angle_grid']
    test_configs = config_dict['test_cfg']

    with open(filename, 'r') as df:
        data = df.readlines()
    if len(data) == 0:
        return None, 0

    n_frame = int(float(data[-1].rstrip().split()[0])) + 1
    dts = {(i, j): [] for i in range(n_frame) for j in range(n_class)}

    for id, line in enumerate(data):
        if line is not None:
            line = line.rstrip().split()
            frameid, class_str, ridx, aidx, conf = line
            frameid = int(frameid)
            classid = get_class_id(class_str, classes)
            ridx = int(ridx)
            aidx = int(aidx)
            conf = float(conf)
            if conf > 1:
                conf = 1
            if conf < model_configs['ols_thres']:
                continue
            rng = rng_grid[ridx]
            agl = agl_grid[aidx]
            if rng > test_configs['rr_max'] or rng < test_configs['rr_min']:
                continue
            if agl > math.radians(test_configs['ra_max']) or agl < math.radians(test_configs['ra_min']):
                continue
            obj_dict = dict(
                id=id + 1,
                frame_id=frameid,
                range=rng,
                angle=agl,
                range_id=ridx,
                angle_id=aidx,
                class_id=classid,
                score=conf
            )
            dts[frameid, classid].append(obj_dict)

    return dts, n_frame


def load_vgg_res(filename, config_dict):
    n_class = config_dict['class_cfg']['n_class']
    classes = config_dict['class_cfg']['classes']
    model_configs = config_dict['model_cfg']
    rng_grid = config_dict['mappings']['range_grid']
    agl_grid = config_dict['mappings']['angle_grid']
    test_configs = config_dict['test_cfg']

    with open(filename, 'r') as df:
        data = df.readlines()

    n_frame = int(float(data[-1].rstrip().split()[0])) + 1
    dts = {(i, j): [] for i in range(n_frame) for j in range(n_class)}

    for id, line in enumerate(data):
        if line is not None:
            line = line.rstrip().split()
            frameid, ridx, aidx, classid, conf = line
            frameid = int(frameid)
            classid = int(classid)
            ridx = int(ridx)
            aidx = int(aidx)
            conf = float(conf)
            if conf > 1:
                conf = 1
            # if conf < 0.5:
            #     continue
            rng = rng_grid[ridx]
            agl = agl_grid[aidx]
            if rng > test_configs['rr_max'] or rng < test_configs['rr_min']:
                continue
            if agl > math.radians(test_configs['ra_max']) or agl < math.radians(test_configs['ra_min']):
                continue
            obj_dict = {'id': id + 1, 'frameid': frameid, 'range': rng, 'angle': agl, 'ridx': ridx, 'aidx': aidx,
                        'classid': classid, 'score': conf}
            dts[frameid, classid].append(obj_dict)

    return dts, n_frame
