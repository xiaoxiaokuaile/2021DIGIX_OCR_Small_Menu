# 2021DIGIX小样本菜单识别赛题 源代码复现步骤

2021DIGIX小样本菜单识别赛题 星光卓越奖 小小快乐队伍 方案及源代码

## 1. 思路简介
我们采用了两阶段的方法来解决该问题，即文本检测和文本识别两步。对于文本检测，我们采用了PSENet[1]。对于文本识别，根据自己初赛阶段的优化经验设计了个多层的CNN+CTC的网络结构。对于小样本特殊字符的识别，采用了数据增强以及finetune等方法进行解决。

## 2. 实现步骤

### 2.1. 预处理

#### 2.1.1 文件夹结构
下面是文件夹的结构示意图，其中checkpoints文件夹包含文本检测算法(detector)以及文本识别算法(recognizer)两部分的程序及模型；official_data文件夹保存的是官方数据集，数据存放结构和示意图一致；output文件夹存放有文本检测算法的输出结果文档test_null.json以及最终提交测评的文档label_special.json;另外两个文档tmp_data和tmp_data_special保存的是文本识别算法需要的数据集。

```
/path/to/
├──checkpoints
│     ├──detector/*
│     ├──recognizer/*
├──official_data
│     ├──train_image_common/*
│     ├──train_image_special/*
│     ├──train_label_common.json
│     ├──train_label_special.json
│     ├──test_image/*
├──output
│     ├──detector_test_output/menu/*
│     ├──test_null.json
│     ├──label_special.json
├──tmp_data
│     ├──recognizer_txts/*
│     ├──recognizer_images/*
├──tmp_data_special
      ├──recognizer_txts/*
      ├──recognizer_images/*
```

#### 2.1.2 环境配置

```
python3.6版本
tensorflow-gpu==1.14.0
torch==1.9.0  其它>=1.1.0版本也可以
torchvision
opencv-python==4.1.0.25
bidict==0.19.0
yacs==0.1.8
Polygon==3.0.9 下载命令是 pip install Polygon3
pyclipper==1.2.1
tqdm
imutils
此外psenet在预测阶段需要用到的opencv包需要用以下命令下载
sudo apt-get install libopencv-dev
其它类型包没有提到的可根据程序运行提安装！
```

运行环境: CPU/GPU (训练需要GPU环境,预测可以CPU/GPU).

### 2.2. 训练文本检测模型

按照上述要求配置好环境以及文件夹结构后，进入/checkpoints/detector/config/resnet50.yaml配置文件中设置路径及超参数，将其中所有/path/to/更换为当前机器所在路径，然后在checkpoints/文件夹下运行:
```
python detector/train.py
```

训练完成后，输出结果为/checkpoints/detector/文件夹下的checkpoint.pth.tar模型文件。

### 2.3. 训练文本识别模型

为了训练文本识别模型，我们首先进行一些预处理。

#### 2.3.1 全量数据集制备

将./checkpoints/recognizer/tools/extract_train_data.py 中294-305行的/path/to/更改为本机文档所在地址，然后在checkpoints/文件夹下运行:

```
python recognizer/tools/extract_train_data.py
```
将./checkpoints/recognizer/tools/from_text_to_label.py中7-8行的/path/to/更改为本机文档所在地址，然后在checkpoints/文件夹下运行:

```
python recognizer/tools/from_text_to_label.py
```

这样文本识别模型的common+special全量数据集已经制作完成，并且生成的chars.txt保存有该数据集所有出现的字符，该文件需要放置在./checkpoints/recognizer/tools/dictionary/中或者打开recognizer/tools/config.py更改其中的路径参数。

#### 2.3.2 special数据集制备

将./checkpoints/recognizer/tools/extract_train_data_special.py中294-301行的/path/to/更改为本机文档所在地址，然后在checkpoints/文件夹下运行:

```
python recognizer/tools/extract_train_data_special.py
```

将./checkpoints/recognizer/tools/from_text_to_label_special.py 中7-8行的/path/to/更改为本机文档所在地址，然后在checkpoints/文件夹下运行:

```
python recognizer/tools/from_text_to_label_special.py 
```

这样文本识别模型finetune的special小样本数据集已经制作完成。

#### 2.3.3 文本识别模型训练

识别模型数据集制备完成后，将./checkpoints/recognizer/train.py中22-25行的/path/to/更改为本机文档所在地址，然后在checkpoints/文件夹下运行:

```
python recognizer/train.py
```

全量数据集训练的文本识别模型CNN_CTC_base_modle.h5保存在/path/to/checkpoint/recognizer/文件夹下，挑选出最优的模型并进行下一步finetune，将./checkpoints/recognizer/train_again.py中23-30行的/path/to/更改为本机文档所在地址，同时将全量数据集中训练得到的最好的模型配置好，然后在checkpoints/文件夹下运行:

```
python recognizer/train_again.py 
```

finetune之后得到最终的文本识别模型CNN_CTC_Final.h5在/path/to/checkpoint/recognizer/文件夹下。

### 2.4. 预测过程
预测分为文本检测网络预测以及文本识别网络预测两部分。
#### 2.4.1. 文本检测模型预测

由于PSENet在预测时候需要用到C++加速，需要做一些预配置工作：

需要将/checkpoints/detector/postprocess/include/pybind11/detail/common.h中的112-113行

```
#include </opt/conda/envs/py36-n/include/python3.6m/Python.h>
#include </opt/conda/envs/py36-n/include/python3.6m/frameobject.h>
#include </opt/conda/envs/py36-n/include/python3.6m/pythread.h>
```

中的地址更改为当前机器中所用(conda)环境下的地址。

然后在/checkpoints文件下运行

```
python detector/predict.py
```
输出结果为/output/文件下的test_null.json文件,它将被用于后面的识别模型预测。
#### 2.4.2. 文本识别模型预测

将/checkpoints/recognizer/predict.py中233-239行的/path/to/更改为本机文档所在地址，然后在checkpoints/文件夹下运行。

```
python recognizer/predict.py 
```

最终得到label_special.json文档保存在./output/文件夹下，该文件即为提交评审文件。

### 2.5. 辅助代码

#### 2.5.1. 文本检测模型检测框绘制

将/checkpoints/detector/Train_img_box_show.py中的地址配置好，选择MODE并运行：

```
python  detector/Train_img_box_show.py
```

即可将训练集或者预测模型预测的测试集文本框绘制出来并保存在/path/to/output/中，方便观察预测效果，有针对的对模型进行调优。

#### 2.5.2. 统计文本识别模型训练集图片宽高比

将/checkpoints/recognizer/tools/H_W_ratio.py中的/path/to/更改为本机地址，并运行：

```
python  recognizer/tools/H_W_ratio.py
```

即可得到相对应的图片集的宽高比统计信息，可以基于此对识别模型网络输入进行合理调整。

## 3. 参考文献

- [1]. Wang, Wenhai, et al. Shape robust text detection with progressive scale expansion network. CVPR. 2019.
- 除了参考论文之外，比赛阶段还搜集参考了大量其他人的代码及论文，一并放在【参考文献】文件夹中，可以根据提示下载查阅。

