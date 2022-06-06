# cv3

#### 车尾识别测距

模型文件有不同训练程度的三个：myhaar.xml，cars3.xml，cascade.xml

图片车尾识别测距在pic_detector.py中

视频车尾识别测距在car_detector.py中，结果在result目录下

#### 道路区域识别

##### 霍夫变换+局部二值化轮廓提取

laneDetection.py

该方法在车辆多，图片模糊场景下效果不佳，已弃用，故没有放到result目录，如有需要可自行运行

##### 基于yolo的深度学习模型

依赖版本为

```
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch
```

且opencv为4.5.1，否则在读取dnn模型时会报错

yolop.onnx为模型文件

car_lane_detector.py对图片进行处理

vedio_lan_detection.py对视频处理

处理结果在result中

#### 道路损害检测

道路损害检测在rddc2020文件夹中，模型装载详情见rddc2020目录的readme文件，测试集放在yolov5/dataset/datasets/road2020/test1/test_images中
