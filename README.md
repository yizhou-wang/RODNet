# RODNet: Radar Object Detection using Cross-Modal Supervision

## Installation

Create a conda environment for RODNet.
```
conda create -n rodnet python=3.8 -y
conda activate rodnet
```

Install pytorch.
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
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
