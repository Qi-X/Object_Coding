# Introduction
This repository is based on our ICME2020 paper *Object-Based Image Coding: A Learning-Driven Revisit*[PDF is available soon](http://pdf.com)
# Usage
We choose PASCAL VOC as our dataset. Our LearntOBIC is trained using the training set of PASCAL VOC 2007 and PASCAL VOC 2012, and tested on the validation set of PASCAL 2012. You can download the [pre-trained model](http://yun.nju.edu.cn/f/0b42a4373d/) to start your training. The object coding model `class object_coding` is built in *new_img_coding.py*
## Training
```
python train.py
```
## Test
```
python test.py
```

# Reference
Deeplab with ResNet34 as backbone: [Code Link](https://github.com/warmspringwinds/pytorch-segmentation-detection)
