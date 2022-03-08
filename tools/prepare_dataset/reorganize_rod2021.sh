# download all zip files and unzip
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
