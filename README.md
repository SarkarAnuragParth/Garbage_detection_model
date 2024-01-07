# Garbage_detection_model
Tensorflow implementation of Transfer learning on MobileNetV2 (pre-trained on ImageNet), for garbage detection. 

## Requirements:
- TensorFlow (version 2.0+)
- OpenCV
- Numpy

## Instructions:
1. Clone the repository using:
```
git clone https://github.com/SarkarAnuragParth/Garbage_detection_model.git
```

2. Your dataset should have the following structure:
```
dataset:
|
|-A:
|  |-No_garbage_image1
|  |-No_garbage_image2...
|-B:
   |-Garbage_present_image1
   |-Garbage_present_image2...
```
Here ```class A ``` will have the label 0 and ```class B``` will have the label 1.

3. Train the model using:
```
python main.py \
--train_path ./path to data directory \
--mode train \
--initial_epochs 10 \
--fine_tune_epochs 10 
```

4. After training, you can test the model using:
```
python main.py \
--train_path ./path to data directory \
--mode test 
```
