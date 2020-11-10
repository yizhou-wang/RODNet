dataset_cfg = dict(
    base_root="/mnt/disk2/CRUW/CRUW_MINI",
    data_root="/mnt/disk2/CRUW/CRUW_MINI/sequences",
    anno_root="/mnt/disk2/CRUW/CRUW_MINI/annotations",
    train=dict(
        seqs=[
            '2019_04_09_BMS1000_PL_NORMAL',
            '2019_04_09_CMS1002_PL_NORMAL',
            '2019_04_09_PMS1000_PL_NORMAL',
            '2019_04_09_PMS3001_PL_NORMAL',
            '2019_05_29_MLMS006_CR_BLUR',
            '2019_05_29_PBMS007_PL_BLUR',
            '2019_09_29_ONRD001_CS_NORMAL',
            '2019_09_29_ONRD002_CS_NORMAL',
            '2019_09_29_ONRD004_HW_NORMAL',
            '2019_10_13_ONRD048_CS_NIGHT'
        ],
    ),
    valid=dict(
        seqs=[],
    ),
    test=dict(
        seqs=[
            '2019_04_09_BMS1000_PL_NORMAL',
        ],
    ),
    demo=dict(
        seqs=[],
    ),
)

model_cfg = dict(
    type='CDC',
    name='rodnet-cdc-win16-wobg',
    max_dets=20,
    peak_thres=0.3,
    ols_thres=0.3,
)

confmap_cfg = dict(
    confmap_sigmas={
        'pedestrian': 15,
        'cyclist': 20,
        'car': 30,
        # 'van': 40,
        # 'truck': 50,
    },
    confmap_sigmas_interval={
        'pedestrian': [5, 15],
        'cyclist': [8, 20],
        'car': [10, 30],
        # 'van': [15, 40],
        # 'truck': [20, 50],
    },
    confmap_length={
        'pedestrian': 1,
        'cyclist': 2,
        'car': 3,
        # 'van': 4,
        # 'truck': 5,
    }
)

train_cfg = dict(
    n_epoch=50,
    batch_size=4,
    lr=0.00001,
    lr_step=5,  # lr will decrease 10 times after lr_step epoches
    win_size=16,
    train_step=1,
    train_stride=8,
    log_step=100,
    save_step=1000,
)
test_cfg = dict(
    test_step=1,
    test_stride=8,
    rr_min=1.0,  # min radar range
    rr_max=20.0,  # max radar range
    ra_min=-60.0,  # min radar angle
    ra_max=60.0,  # max radar angle
)
