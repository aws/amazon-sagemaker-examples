data_folder="coco"
mkdir -p $data_folder
cd $data_folder
COCO_IMAGES_URL=http://images.cocodataset.org/zips/
TRAIN_ZIP=train2017.zip
VAL_ZIP=val2017.zip
TEST_ZIP=test2017.zip
COCO_ANNO_URL=http://images.cocodataset.org/annotations/
TRAIN_VAL_ANNO_ZIP=annotations_trainval2017.zip
wget $COCO_IMAGES_URL$TRAIN_ZIP
wget $COCO_IMAGES_URL$VAL_ZIP
wget $COCO_IMAGES_URL$TEST_ZIP
wget $COCO_ANNO_URL$TRAIN_VAL_ANNO_ZIP
unzip $TRAIN_ZIP
unzip $VAL_ZIP
unzip $TEST_ZIP
unzip $TRAIN_VAL_ANNO_ZIP