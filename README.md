# RODNet: Radar Object Detection using Cross-Modal Supervision

This is the official implementation of our RODNet paper at WACV 2021. 

[[Paper]](https://openaccess.thecvf.com/content/WACV2021/html/Wang_RODNet_Radar_Object_Detection_Using_Cross-Modal_Supervision_WACV_2021_paper.html)
[[Dataset]](https://www.cruwdataset.org)

![RODNet Overview](./assets/images/overview.jpg?raw=true)

Please cite our WACV 2021 paper if this repository is helpful for your research:
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
## Installation

Create a conda environment for RODNet. Tested under Python 3.6, 3.7, 3.8.
```
conda create -n rodnet python=3.* -y
conda activate rodnet
```

Install pytorch.
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

Install `cruw-devkit` package. 
Please refer to [`cruw-devit`](https://github.com/yizhou-wang/cruw-devkit) repository for detailed instructions.
```
git clone https://github.com/yizhou-wang/cruw-devkit.git
cd cruw-devkit
pip install -e .
cd ..
```

Setup RODNet package.
```
pip install -e .
```

## Prepare data for RODNet

```
python tools/prepare_dataset/prepare_data.py \
        --config configs/<CONFIG_FILE> \
        --data_root <DATASET_ROOT> \
        --split train,test \
        --out_data_dir data/<DATA_FOLDER_NAME>
```

## Train models

```
python tools/train.py --config configs/<CONFIG_FILE> \
        --data_dir data/<DATA_FOLDER_NAME> \
        --log_dir checkpoints/
```

## Inference

```
python tools/test.py --config configs/<CONFIG_FILE> \
        --data_dir data/<DATA_FOLDER_NAME> \
        --checkpoint <CHECKPOINT_PATH> \
        --res_dir results/
```
