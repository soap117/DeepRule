# DeepRule
Compete code of DeepRule
## Getting Started
Please first install [Anaconda](https://anaconda.org) and create an Anaconda environment using the provided package list.
```
 conda create  --DeepRule python-course --file DeepRule.txt
```

After you create the environment, activate it.
```
source activate DeepRule
```

Our current implementation only supports GPU so you need a GPU and need to have CUDA installed on your machine.

### Compiling Corner Pooling Layers
You need to compile the C++ implementation of corner pooling layers. 
```
cd <CornerNet dir>/models/py_utils/_cpools/
python setup.py build_ext --inplace
```

### Compiling NMS
You also need to compile the NMS code (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/cpu_nms.pyx) and [Soft-NMS](https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx)).
```
cd <CornerNet dir>/external
make
```

### Installing MS COCO APIs
You also need to install the MS COCO APIs.
```
cd <CornerNet dir>/data
git clone git@github.com:cocodataset/cocoapi.git coco
cd <CornerNet dir>/data/coco/PythonAPI
make
```

### Downloading CHARTEX Data
- Unzip the file to the data path
### Downloading Trained File
- [data link](https://drive.google.com/file/d/1qtCLlzKm8mx7kQOV1criUbqcGnNh58Rr/view?usp=sharing)
- Unzip the file to current root path 
## Training and Evaluation
To train and evaluate a network, you will need to create a configuration file, which defines the hyperparameters, and a model file, which defines the network architecture. The configuration file should be in JSON format and placed in `config/`. Each configuration file should have a corresponding model file in `models/`. i.e. If there is a `<model>.json` in `config/`, there should be a `<model>.py` in `models/`. There is only one exception which we will mention later.
The cfg file names of our proposed modules are as follows:

Bar: CornerNetPureBar

Pie: CornerNetPurePie

Line: CornerNetLine

Query: CornerNetLineClsReal

Cls: CornerNetCls

To train a model:
```
python train.py --cfg_file <model> --data_dir <data path> 
e.g. 
python train_chart.py --cfg_file CornerNetBar --data_dir /home/data/bardata(1031)
```

To use the trained model as a web server pipeline:
```
python manage.py runserver 8800
```
Access localhost:8800 to interact.

If you want to test batch of data directly, here you have to pre-assign the type of charts.
```
python test_pipe_type_cloud.py --image_path <image_path> --save_path <save_path> --type <type>
e.g.
python test_pipe_type_cloud.py --image_path /data/bar_test --save_path save --type Bar
```
