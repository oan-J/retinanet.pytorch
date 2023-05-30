# retinanet.pytorch


## Setup
### Environment Preparation
- if you work on a MacOS system
```
conda install -c anaconda tk
```
- necessary python packages
```
pip install -r requirements.txt
```
### Model
The project will automatically download **resnet50-19c8e357**.
### Datasets
Download PASVAL VOC 2007 into allVOCdata/VOCdevkit/

If you want to use other PASVAL VOC datasets(such as VOC 2012), you should change the path accordingly in the divide.py file after downloading the dataset.
```
Annotations = get_file_index('allVOCdata/VOCdevkit/VOC2012/Annotations', '.xml')
```
```
JPEGfiles = get_file_index('allVOCdata/VOCdevkit/VOC2012/JPEGImages','.jpg')  
```
## Train
Befor training
- run the divide.py file, this will generate 3 csv files and divide the dataset into trainset and testset.
    - class.csv
    - train.csv
    - val.csv
```
python3 divide.py
```
Train
```
python3 train.py --dataset csv --csv_train train.csv  --csv_classes class.csv  --csv_val val.csv
```
