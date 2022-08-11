#!/bin/bash
git clone https://github.com/yizhou-wang/RODNet.git
conda create -n rodnet python=3.* -y
conda init bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate rodnet
conda -y install pytorch=1.4 torchvision cudatoolkit=10.1 -c pytorch
git clone https://github.com/yizhou-wang/cruw-devkit.git
cd cruw-devkit
pip install .
cd ..
cd RODNet
pip install -e .

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-87V3rBgJHU4HpCtAnu6zhxSMjMTpE5N' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-87V3rBgJHU4HpCtAnu6zhxSMjMTpE5N" -O TRAIN_RAD_H.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1cTLfwxNw62Km8yNVmKzzHWi2tdaUeVDW' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1cTLfwxNw62Km8yNVmKzzHWi2tdaUeVDW" -O TRAIN_CAM_0.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rS5eO_qcUPWaYK7NRt7d3ynyjDnmyIYd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1rS5eO_qcUPWaYK7NRt7d3ynyjDnmyIYd" -O TRAIN_RAD_H_ANNO.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Mc8_WSUKJvP8_WgmGSuu5D0RJ0TPZViB' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Mc8_WSUKJvP8_WgmGSuu5D0RJ0TPZViB" -O CAM_CALIB.zip && rm -rf /tmp/cookies.txt


unzip TRAIN_RAD_H.zip
unzip TRAIN_CAM_0.zip
unzip TEST_RAD_H.zip
unzip TRAIN_RAD_H_ANNO.zip
unzip CAM_CALIB.zip

# make folders for data and annotations
mkdir sequences
mkdir annotations

# rename unzipped folders
mv TRAIN_RAD_H sequences/train
mv TRAIN_CAM_0 train
mv TEST_RAD_H sequences/test
mv TRAIN_RAD_H_ANNO annotations/train

# merge folders and remove redundant
rsync -av train/ sequences/train/
rm -r train
