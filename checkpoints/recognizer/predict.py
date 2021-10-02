# -*- coding=utf-8 -*-
import argparse
import os
import json
import cv2
import sys
import math
import numpy as np

from itertools import groupby
from enum import Enum


basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(basedir)

from recognizer.models.crnn_model import crnn_model_based_on_densenet_crnn_time_softmax_activate
from recognizer.tools.config import config
from recognizer.tools.utils import get_chinese_dict

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class ModelType(Enum):
    DENSENET_CRNN_TIME_SOFTMAX = 0

# =================================== 求两点间距离 ========================================
def getDist_P2P(Point0, PointA):
    distance = math.pow((Point0[0] - PointA[0]), 2) + math.pow((Point0[1] - PointA[1]), 2)
    distance = math.sqrt(distance)
    return distance
# =============================== 计算透视变换参数矩阵 =======================================
def cal_perspective_params(img, points):
    img_size = (np.int32(getDist_P2P(points[0],points[1])),np.int32(getDist_P2P(points[0],points[2])))
    src = np.float32(points)
    # 透视变换的四个点 左上,右上,左下,右下
    dst = np.float32([[0, 0], [img_size[0], 0],[0, img_size[1]], [img_size[0], img_size[1]]])
    # 透视矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    # 透视逆矩阵
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    return M, M_inverse,img_size

# 输入原始图片 + 点阵求最小外接矩形并进行仿射变换,得到修正后的图
def Image_out(src_image,src_point_list):
    # 求外接矩形
    src_point_list = np.array(src_point_list)
    _min_area_box = cv2.minAreaRect(src_point_list.astype(np.float32))
    # 将外接矩形转化为4个点坐标
    _min_area_box = np.array(cv2.boxPoints(_min_area_box), dtype=np.int32)
    # 得到左上,右上,左下,右下四个点坐标
    sorted_point_list_by_x = sorted(_min_area_box, key=lambda x: x[0])
    left_points = sorted_point_list_by_x[:2]
    left_points_sort = sorted(left_points, key=lambda x: x[1])
    right_points = sorted_point_list_by_x[2:]
    right_points_sort = sorted(right_points, key=lambda x: x[1])
    points = [left_points_sort[0], right_points_sort[0], left_points_sort[1], right_points_sort[1]]
    # 计算透视变换参数矩阵
    M, M_inverse, img_size = cal_perspective_params(src_image, points)
    # 透视变换
    crop_image = cv2.warpPerspective(src_image, M, img_size)
    return crop_image,img_size

def load_model(model_type, weight):
    if model_type == ModelType.DENSENET_CRNN_TIME_SOFTMAX:
        base_model, _ = crnn_model_based_on_densenet_crnn_time_softmax_activate()
        base_model.load_weights(weight)
    else:
        raise ValueError('parameter model_type error.')
    return base_model

def img_predict(img,base_model,input_width, input_height, input_channel):
    img_height, img_width = img.shape[0:2]
    new_image = np.ones((input_height, input_width, input_channel), dtype='uint8')
    new_image[:] = 255
    # 若为灰度图，则扩充为彩色图
    if input_channel == 1:
        img = np.expand_dims(img, axis=2)
    new_image[:, :img_width, :] = img
    image_clip = new_image

    text_image = np.array(image_clip, 'f') / 127.5 - 1.0
    text_image = np.reshape(text_image, [1, input_height, input_width, input_channel])
    y_pred = base_model.predict(text_image)

    return y_pred

# 通过图片预测文本
def predict(image, input_shape, base_model):
    # 32,560,3
    input_height, input_width, input_channel = input_shape
    # 缩放比例
    scale = image.shape[0] * 1.0 / input_height
    # ================================== 改动 ===========================================
    if scale==0:
        scale = -1

    image_width = int(image.shape[1] // scale)
    if image_width <= 0:
        return ''
    # resize大小
    image = cv2.resize(image, (image_width, input_height))
    image_height, image_width = image.shape[0:2]
    # =============== 得到真实图片宽高比 ==============
    w_h_ratio = image_width/image_height
    # ==============================================
    # 如果图像宽度小于560，对多出来部分补255
    if image_width <= input_width:
        y_pred = img_predict(image, base_model, input_width, input_height, input_channel)
        y_pred = y_pred[:, :, :]
    elif w_h_ratio<=35:
        # 切两段分别预测
        # 切片1
        imgg1 = image[:, :int(image.shape[1] / 2), :]
        y_pred1 = img_predict(imgg1, base_model, input_width, input_height, input_channel)
        y_pred1 = y_pred1[:, :, :]
        # 切片2
        imgg2 = image[:, int(image.shape[1] / 2):, :]
        y_pred2 = img_predict(imgg2, base_model, input_width, input_height, input_channel)
        y_pred2 = y_pred2[:, :, :]
        # 两个切片预测结果合并
        y_pred = np.concatenate((y_pred1, y_pred2), axis=1)
    elif w_h_ratio<=53:
        # 切片1
        imgg1 = image[:, :int(image.shape[1] / 3), :]
        y_pred1 = img_predict(imgg1,base_model, input_width, input_height, input_channel)
        y_pred1 = y_pred1[:, :, :]
        # 切片2
        imgg2 = image[:, int(image.shape[1] / 3):int(image.shape[1] * 2 / 3), :]
        y_pred2 = img_predict(imgg2, base_model, input_width, input_height, input_channel)
        y_pred2 = y_pred2[:, :, :]
        # 切片3
        imgg3 = image[:, int(image.shape[1] * 2 / 3):, :]
        y_pred3 = img_predict(imgg3, base_model, input_width, input_height, input_channel)
        y_pred3 = y_pred3[:, :, :]
        # 两个切片预测结果合并
        y_pred = np.concatenate((y_pred1, y_pred2, y_pred3), axis=1)
    else:
        # 切片1
        imgg1 = image[:, :int(image.shape[1] / 3), :]
        image_clip1 = cv2.resize(imgg1, (input_width, input_height))
        text_image1 = np.array(image_clip1, 'f') / 127.5 - 1.0
        text_image1 = np.reshape(text_image1, [1, input_height, input_width, input_channel])
        y_pred1 = base_model.predict(text_image1)
        y_pred1 = y_pred1[:, :, :]
        # 切片2
        imgg2 = image[:, int(image.shape[1] / 3):int(image.shape[1] * 2 / 3), :]
        image_clip2 = cv2.resize(imgg2, (input_width, input_height))
        text_image2 = np.array(image_clip2, 'f') / 127.5 - 1.0
        text_image2 = np.reshape(text_image2, [1, input_height, input_width, input_channel])
        y_pred2 = base_model.predict(text_image2)
        y_pred2 = y_pred2[:, :, :]
        # 切片3
        imgg3 = image[:, int(image.shape[1] * 2 / 3):, :]
        image_clip3 = cv2.resize(imgg3, (input_width, input_height))
        text_image3 = np.array(image_clip3, 'f') / 127.5 - 1.0
        text_image3 = np.reshape(text_image3, [1, input_height, input_width, input_channel])
        y_pred3 = base_model.predict(text_image3)
        y_pred3 = y_pred3[:, :, :]
        # 三个切片预测结果合并
        y_pred = np.concatenate((y_pred1, y_pred2, y_pred3), axis=1)

    char_list = list()
    # 得到最大值索引,即得到大小280字符索引列表
    pred_text = list(y_pred.argmax(axis=2)[0])
    # 将预测的数字转换为字符
    for index in groupby(pred_text):
        # 若index不为282,空键
        if index[0] != config.num_class - 1:
            char_list.append(character_map_table[str(index[0])])

    return u''.join(char_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--char_path', type=str, default='recognizer/tools/dictionary/chars.txt')
    parser.add_argument('--model_path', type=str,default='/opt/menu_data/checkpoints/recognizer/CNN_CTC_Final.h5')
    # detector output --- .json
    parser.add_argument('--null_json_path', type=str,default='/opt/menu_data/output/test_null.json')
    # test image path --- have Rectangular box
    parser.add_argument('--test_image_path', type=str,default='/opt/menu_data/output/detector_test_output/menu')
    parser.add_argument('--submission_path', type=str,default='/opt/menu_data/output/label_special.json')
    opt = parser.parse_args()

    character_map_table = get_chinese_dict(opt.char_path)
    input_shape = (32, 560, 3)
    model = load_model(ModelType.DENSENET_CRNN_TIME_SOFTMAX, opt.model_path)
    print('load model done.')

    # 测试集null_josn路径
    test_label_json_file = opt.null_json_path
    test_image_root_path = opt.test_image_path
    with open(test_label_json_file, 'r', encoding='utf-8') as in_file:
        label_info_dict = json.load(in_file)
        for idx, info in enumerate(label_info_dict.items()):
            image_name, text_info_list = info
            src_image = cv2.imread(os.path.join(test_image_root_path, image_name.split('.')[0] + '.JPEG'))
            # print(os.path.join(test_image_root_path, image_name))
            print('process: {:3d}/{:3d}. image: {}'.format(idx + 1, len(label_info_dict.items()), image_name))
            for index, text_info in enumerate(text_info_list):
                src_point_list = text_info['points']
                # ========================== 外接矩形+仿射透视 =======================
                crop_image, img_size = Image_out(src_image, src_point_list)
                # ====================== 如果竖直方向文本,右转90度 ====================
                if 2 * img_size[0] < img_size[1]:
                    print('Vertical picture!')
                    crop_image = np.rot90(crop_image, -1)
                # =================================================================
                rec_result = predict(crop_image, input_shape, model)
                # 图片切片的预测label
                text_info['label'] = rec_result

    save_label_json_file = opt.submission_path
    with open(save_label_json_file, 'w') as out_file:
        out_file.write(json.dumps(label_info_dict))

'''bash
    python recognizer/predict.py
'''
