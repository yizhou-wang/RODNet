# RODNet: Radar Object Detection Network

This is the official implementation of our RODNet papers 
at [WACV 2021](https://openaccess.thecvf.com/content/WACV2021/html/Wang_RODNet_Radar_Object_Detection_Using_Cross-Modal_Supervision_WACV_2021_paper.html) 
and [IEEE J-STSP 2021](https://ieeexplore.ieee.org/abstract/document/9353210). 

[[Arxiv]](https://arxiv.org/abs/2102.05150)
[[Dataset]](https://www.cruwdataset.org)
[[ROD2021 Challenge]](https://codalab.lisn.upsaclay.fr/competitions/1063)
[[Presentation]](https://youtu.be/UZbxI4o2-7g)
[[Demo]](https://youtu.be/09HaDySa29I)

![RODNet Overview](./assets/images/overview.jpg?raw=true)

Please cite our paper if this repository is helpful for your research:
```
@inproceedings{wang2021rodnet,
  author={Wang, Yizhou and Jiang, Zhongyu and Gao, Xiangyu and Hwang, Jenq-Neng and Xing, Guanbin and Liu, Hui},
  title={RODNet: Radar Object Detection Using Cross-Modal Supervision},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  month={January},
  year={2021},
  pages={504-513}
}
```
```
@article{wang2021rodnet,
  title={RODNet: A Real-Time Radar Object Detection Network Cross-Supervised by Camera-Radar Fused Object 3D Localization},
  author={Wang, Yizhou and Jiang, Zhongyu and Li, Yudong and Hwang, Jenq-Neng and Xing, Guanbin and Liu, Hui},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  volume={15},
  number={4},
  pages={954--967},
  year={2021},
  publisher={IEEE}
}
```

## Installation and Download of Data
(Verify you are at CUDA 10.X)

docker build -t rodnet-docker ./

docker run --rm -it --gpus all -v /:/workspace/ -p 52713:52713 --name rodnet-docker rodnet-docker bash

Run inside your docker
./install_rodnet

conda activate rodnet

Thats it.

## Prepare data for RODNet

Download [ROD2021 dataset](https://www.cruwdataset.org/download#h.mxc4upuvacso). 
Follow [this script](https://github.com/yizhou-wang/RODNet/blob/master/tools/prepare_dataset/reorganize_rod2021.sh) to reorganize files as below.

```
data_root
  - sequences
  | - train
  | | - <SEQ_NAME>
  | | | - IMAGES_0
  | | | | - <FRAME_ID>.jpg
  | | | | - ***.jpg
  | | | - RADAR_RA_H
  | | |   - <FRAME_ID>_<CHIRP_ID>.npy
  | | |   - ***.npy
  | | - ***
  | | 
  | - test
  |   - <SEQ_NAME>
  |   | - RADAR_RA_H
  |   |   - <FRAME_ID>_<CHIRP_ID>.npy
  |   |   - ***.npy
  |   - ***
  | 
  - annotations
  | - train
  | | - <SEQ_NAME>.txt
  | | - ***.txt
  | - test
  |   - <SEQ_NAME>.txt
  |   - ***.txt
  - calib
```

Convert data and annotations to `.pkl` files.
```commandline
python tools/prepare_dataset/prepare_data.py \
        --config configs/<CONFIG_FILE> \
        --data_root <DATASET_ROOT> \
        --split train,test \
        --out_data_dir data/<DATA_FOLDER_NAME>
```

## Train models

```commandline
python tools/train.py --config configs/<CONFIG_FILE> \
        --data_dir data/<DATA_FOLDER_NAME> \
        --log_dir checkpoints/
```

## Inference

```commandline
python tools/test.py --config configs/<CONFIG_FILE> \
        --data_dir data/<DATA_FOLDER_NAME> \
        --checkpoint <CHECKPOINT_PATH> \
        --res_dir results/
```
