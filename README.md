# retinanet.pytorch
<h3>
<strong>
<font color="green"> 
Abstraction: Train and test  RetinaNet on VOC datasets.
</font>
</strong>
</h3>

Warning: please make sure your dataset(the data and the path) has no problem, or else you can't run the project successfully.

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
- Run the generate_data_path.py.
It will generate a path for you to put the datasets.
```commandline
cd code
python3 generate_data_path.py
```

- Download PASVAL VOC 2007 into ../Retinanet/data/allVOCdata/VOCdevkit/
In the end, the dataset will look like this.
```
.
└── data
    └── allVOCdata
        └── VOCdevkit
            └── VOC2007
                ├── Annotations
                ├── ImageSets
                ├── JPEGImages
                ├── SegmentationClass
                └── SegmentationObject

```
You can always switch to another directory, remember to change all the other paths correspondingly.

If you want to use other PASVAL VOC datasets(such as VOC 2012), you should change the path accordingly in the divide.py file after downloading the dataset.
```
Annotations = get_file_index('../Retinanet/data/allVOCdata/VOCdevkit/VOC2012/Annotations', '.xml')
```
```
JPEGfiles = get_file_index('../Retinanet/data/allVOCdata/VOCdevkit/VOC2012/JPEGImages','.jpg')  
```
## Train
Befor training
- run the divide.py file, this will generate 3 csv files and divide the dataset into trainset and testset.
    - class.csv
    - train.csv
    - val.csv
```
cd code
python3 divide.py
```
Train
```
cd code
python3 train.py --dataset csv --csv_train train.csv  --csv_classes class.csv  --csv_val val.csv
```
## Test
```
cd code
python3 csv_validation.py --csv_annotations_path path/to/annotations.csv --model_path path/to/model.pt --images_path path/to/images_dir --class_list_path path/to/class_list.csv (optional) iou_threshold iou_thres (0<iou_thresh<1)
```
## Visualize

To visualize the network detection, use `visualize.py`:

```
cd code
(python3 visualize.py --dataset csv --csv_path ../csv --model <path/to/model.pt>)
python3 visualize.py --dataset csv --csv_classes class.csv --csv_val val.csv --model ../out/model_final.pt
```
This will visualize bounding boxes on the validation set. To visualise with a CSV dataset, use:

```
cd code
python3 visualize.py --dataset csv --csv_classes <path/to/train/class_list.csv>  --csv_val <path/to/val_annots.csv> --model <path/to/model.pt>
```



## Credit
https://github.com/yhenon/pytorch-retinanet