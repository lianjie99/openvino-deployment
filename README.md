# OpenVINO deployment guide

OpenVINO deployment guide. Please make sure that `openvino-2021.4` is installed <br />
If openvino haven't installed, please follow the following guide: <br />

https://docs.openvino.ai/latest/openvino_docs_install_guides_installing_openvino_linux.html#step-1-install-the-intel-distribution-of-openvino-toolkit-core-components

### **Make sure environment is setup first**
```shell
cd /opt/intel/openvino_2021/
source bin/setupvars.sh
```
Expected output if the environment is setup successfully

![openvino2021_environment](/uploads/e7df4db793d2c32106497130a2990e40/openvino2021_environment.PNG)

### **This guide consists of three major parts:**<br />
[**Setting up the model (IR format)**](#setting-up-the-model-ir-format)
1) [Convert pretrained custom model to obtain IR format (.xml + .bin)](#converting-a-pretrained-model-in-openvino)
2) [Download and use available model provided in OpenVINO](#download-available-model-in-openvino)
3) [Build with YOLOv5s](#build-with-yolov5s-pytorch-yolov5s-onnx-file-bin-xml)

[**Running the model with OpenVINO inference**](#running-the-model-with-openvino-inference)
1) [Classification](#classification)
2) [Object Detection](#object-detection)

[**Running the model with OpenVINO Model Server (OVMS)**](#running-the-model-with-openvino-model-server-ovms)
1) [Install Docker](#1-install-docker)
2) [Install Requirements](#2-install-requirements)
2) [Create Container](#3-create-container)
3) [Deploy !](#4-deploy)


# **Setting up the model (IR format)**
## **Converting a pretrained model in OpenVINO**
To refer the complete converting TF model documentation:

<https://docs.openvino.ai/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html>


### **1) Install prerequisites for TF2**
```shell
cd /opt/intel/openvino_2021/deployment_tools/model_optimizer/install_prerequisites
./install_prerequisites_tf2.sh
```

### **2) Converting the model (OpenVINO expects BGR format by default)**
To refer the documentation for specific parameters when doing conversion:

<https://docs.openvino.ai/2020.1/_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html>

OR just pass --help argument into the `mo_tf.py` script

**_NOTE:_**
OpenVINO conversion will fail if the model consists of certain augmentation layer

#### **Option 1:**
Convert **pb** model into IR format (convert TF model > pb file > IR format (.xml + .bin))

Example:
```shell
cd /opt/intel/openvino_2021/deployment_tools/model_optimizer
python3 mo_tf.py --input_model ~/Desktop/LianJie-Build/models_before_convert/mobilenetv1.pb --output_dir ~/Desktop/LianJie-Build/models --input_shape [1,256,256,3] --reverse_input_channels
```

#### **Option 2:**
Convert **SavedModel** model into IR format (convert TF model > SavedModel dir > IR format (.xml + .bin))

Example:
```shell
cd /opt/intel/openvino_2021/deployment_tools/model_optimizer
python3 mo_tf.py --output_dir ~/Desktop/LianJie-Build/models --input_shape [1,256,256,3] --saved_model_dir ~/Desktop/LianJie-Build/models_before_convert/effnetb0_savedmodel --reverse_input_channels
```


## **Download available model in OpenVINO**
### **Intel model**
The list of available intel models can be obtained from here: <br />
https://docs.openvino.ai/latest/omz_models_group_intel.html <br />
### **Download the model into a folder**
Example of downloading `person-detection-0201`
```shell
cd /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader
python3 downloader.py --name person-detection-0201 --output_dir ~/Desktop/LianJie-Build/downloaded_models
```
From the above example, you should be able to see the downloaded IR model as follows:
```
downloaded_models
│       
└───intel
   │
   └───person-detection-0201
        │  
        └──FP16
        |    |    
        |    └──person-detection-0201.bin
        |    └──person-detection-0201.xml
        |
        └──FP16-INT8
        |    |
        |    └──person-detection-0201.bin
        |    └──person-detection-0201.xml
        |
        └──FP32
            |
            └──person-detection-0201.bin
            └──person-detection-0201.xml
```

### **Public model**
The list of available public models can be obtained from here: <br />
https://docs.openvino.ai/latest/omz_models_group_public.html <br />
### **1) Download the model into a folder**
**_NOTE:_**
In most cases, public model is more troublesome than intel model as it exists as yaml or other format instead of IR format. Hence, further model conversion is required.

Example of downloading `yolo-v4-tiny-tf`:
```shell
cd /opt/intel/openvino_2021/deployment_tools/open_model_zoo/tools/downloader
python3 downloader.py --name yolo-v4-tiny-tf --output_dir ~/Desktop/LianJie-Build/downloaded_models
```
From the above example, you should be able to see the downloaded yolov4 tiny folder as follows:

```
downloaded_models
│       
└───public
   │
   └───yolo-v4-tiny-tf
```
### **2) Convert the public model into a pb format**
**_NOTE:_**

- Each public model has its own conversion method. 
- For example, the `README.md` in <br /> `/opt/intel/openvino_2021/deployment_tools/open_model_zoo/models/public/yolo-v4-tiny-tf` consists of the instruction on how to convert the `yolov4-tiny` into pb format

```shell
cd /opt/intel/openvino_2021/deployment_tools/open_model_zoo/models/public/yolo-v4-tiny-tf
python3 pre-convert.py ~/Desktop/LianJie-Build/downloaded_models/public/yolo-v4-tiny-tf ~/Desktop/LianJie-Build/models_before_convert
```

From the above example, you should be able to see the downloaded yolov4 tiny pb as follows:
```
models_before_convert
│       
└───yolo-v4-tiny.h5
|
└───yolo-v4-tiny.pb
```

### **3) Convert the pb model into IR format**
This step is exactly the same as <br />
[converting pb format into IR format](#option-1)<br />
**_Note:_** Some model is using `mo.py` instead of `mo_tf.py` as it is according to its model architecture (refer to its `README.md` )

## **Build with YOLOv5s (PyTorch YOLOv5s > onnx file > bin + xml)**
**_NOTE:_** TF YOLOv5s cannot be converted into IR format because of the combined_non_max_suppression layer within it. For more info, refer:
1) https://github.com/openvinotoolkit/openvino/issues/5875
2) https://github.com/openvinotoolkit/openvino/issues/3165

### Follow the following github repo procedure to build a YOLOv5s from PyTorch (ultralytics) directly <br />
https://github.com/violet17/yolov5_demo

### Convert into .xml + .bin
To view the output Conv layer, please use <https://netron.app/> to get the output name layer
```shell
python3 /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py  --input_model yolov5s.onnx --model_name yolov5s -s 255 --reverse_input_channels --output Conv_305,Conv_359,Conv_251 --output_dir ~/Desktop/LianJie-Build/models
```
**_Note:_** It is using `mo.py` instead of `mo_tf.py` as it is a PyTorch model instead of TF


# **Running the model with OpenVINO inference**
## **Classification**

- Use `classification_evaluate.py` provided in the repo to run the classification

- The original demo code can be found in <br /> `/opt/intel/openvino_2021/inference_engine/samples/python/classification_sample_async` <br />

- Refer the --help argument for specific parameters when doing conversion <br />

Example running with `EfficientNetB0` using CPU:
```shell
cd ~/Desktop/LianJie-Build
python3 classification_evaluate.py -i ~/Desktop/LianJie-Build/evaluate_data/Train_mixed/EmptyBed -m ~/Desktop/LianJie-Build/models/effnetb0_FP16.xml -d CPU
```
If want to run with GPU change `-d` to `GPU` instead of CPU or use `MULTI:CPU,GPU` for both

## **Object Detection**
### YOLOv1,YOLOv2,YOLOv3,YOLOv4,SSD,RetinaNet etc 
- Use `object_detection_demo.py` provided in the (demos folder) to run the object detection<br />

**_NOTE:_** The entire demos folder is required as there are several modules needed in the common folder<br />

- The original demo code can be found in <br /> `/opt/intel/openvino_2021/inference_engine/demos/object_detection_demo/python<br />`

- Refer the --help argument for specific parameters when doing conversion <br />

Example running with `person-detection-0201` using CPU:

```shell
cd ~/Desktop/LianJie-Build/demos/object_detection_demo/python
python3 object_detection_demo.py -m ~/Desktop/LianJie-Build/downloaded_models/intel/person-detection-0201/FP16/person-detection-0201.xml -at ssd -i ~/Desktop/LianJie-Build/Data/counting_people.mp4 -o ~/Desktop/LianJie-Build/Data/counting_person_detection_0201.avi --input_size 384 384 -d CPU
```

**_NOTE:_** This object detection script **do not** support certain architecture such as yolov5. The detailed supported frameworks is stated in the `README.md` from the original demo script

### YOLOv5

- For yolov5s, just run the `yolov5_inference.py` will do<br />

Example running yolov5s using CPU:
```shell
cd ~/Desktop/LianJie-Build
python3 yolov5_inference.py -i Data/group2.jpg -m models/yolov5s.xml -d CPU
```
# **Running the model with OpenVINO Model Server (OVMS)**
OpenVINO quick start guide (optional): <br />
https://github.com/openvinotoolkit/model_server/blob/main/docs/ovms_quickstart.md

## **1) Install Docker**
Follow this link to install docker <br />
https://docs.docker.com/engine/install/ubuntu/


## **2) Install Requirements**
Example:
```shell
cd ~/Desktop/LianJie-Build/OVMS
pip install -r client_requirements.txt
```

## **3) Create Container**
### Go to your desired working directory that contains model directory
Example:
```shell
cd ~/Desktop/LianJie-Build/OVMS
```
### Create docker container with specific docker image

Before creating a container, make sure that the model is in the correct directory format:<br />
Example: <br />
```
models
|              
└─── model1
    │       
    └───1
        │       
        └─── effnetb0.xml
        │       
        └─── effnetb0.bin
```

Creating a container after the model is in the correct directory format:<br />
Example: <br />
**CPU:**
```shell
sudo docker run -d -v $(pwd)/models:/models/ -p 9000:9000 openvino/model_server:latest --model_path /models/model1/ --model_name effnetb0 --port 9000 --shape auto
```

**GPU:** (Ubuntu 20.04)
```shell
sudo docker run -d --rm -it  --device=/dev/dri --group-add=$(stat -c "%g" /dev/dri/render*) -u $(id -u):$(id -g) -v $(pwd)/models:/models/ -p 9002:9002 openvino/model_server:latest-gpu --model_path /models/model1 --model_name effnetb0-GPU --port 9002 --target_device GPU
```
**_Note:_**
In practice, we should be able to switch into GPU mode by changing the docker image into latest-gpu and target_device to GPU. However,there are certain reasons of Ubuntu version (20.04) in deploying GPU for OpenCL which is stated below:<br />
https://github.com/openvinotoolkit/docker_ci/blob/master/configure_gpu_ubuntu20.md


## **4) Deploy!**

- For OVMS deployment, the python script expects the input and output layer name from the model.
- To do so, please use the <https://netron.app/> to obtain specific layer's name

Example of input layer name and output layer name:

input:<br />
![input_name](/uploads/3cd8777aab62aa656b9277e0a3ca7feb/input_name.PNG)<br />
output:<br />
![output_name](/uploads/1d5a7bfc290c18cf6ebdeac5e0b10c17/output_name.PNG)<br />

`input_name` = "input_1"<br />
`output_name` = "StatefulPartitionedCall/model/dense_2/Softmax"<br />

From the script, search for `request.inputs[input_name]` and `result.outputs[output_name]`to modify accordingly

**_NOTE:_** The deployment scripts require `client_utils.py` for the statistic analysis<br />
### Classification

- Use `classification_OVMS.py` provided in the (OVMS folder) to run the classification <br />

- Please modify for specific classes and the targeted class when making evaluation <br />

- Use --help to obtain the required parameters **(model name and port number is important)**
```shell
cd ~/Desktop/LianJie-Build/OVMS
python3 classification_OVMS.py --input_images_dir ~/Desktop/LianJie-Build/evaluate_data/Train_mixed/EmptyBed
```

### Object Detection

- Use `object_detection_OVMS.py` provided in the (OVMS folder) to run the object detection<br />

- Use --help to obtain the required parameters **(model name and port number is important)**
```shell
cd ~/Desktop/LianJie-Build/OVMS
python3 object_detection_OVMS.py --isvid True
```


# Common error

- if the permission denied occur when accessing multiple containers at once, follow this:<br />
https://stackoverflow.com/questions/47854463/docker-got-permission-denied-while-trying-to-connect-to-the-docker-daemon-socke


- if GPU version error for OpenVINO inference:<br />
https://github.com/intel/compute-runtime/issues/325


- MYRIAD issue:<br />
https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/ERROR-Can-not-init-Myriad-device-NC-ERROR/m-p/1226832#M21382


