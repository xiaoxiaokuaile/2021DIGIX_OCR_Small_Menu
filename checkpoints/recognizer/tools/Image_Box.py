# Learner: 王振强
# Learn Time: 2021/9/18 18:18
# -*- coding=utf-8 -*-
import argparse
import os
import json
import numpy as np
import cv2
import math
import random
import imutils

global_image_num = 0
char_set = set()


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

# 原始图片路径、原始图片json文件、保存图片路径、保存txt路径
def extract_train_data(src_image_root_path, src_label_json_file, save_image_path, save_txt_path,enhance_num=2):
    global global_image_num, char_set
    with open(src_label_json_file, 'r', encoding='utf-8') as in_file:
        label_info_dict = json.load(in_file)
        with open(os.path.join(save_txt_path, 'train.txt'), 'a', encoding='utf-8') as out_file:
            for image_name, text_info_list in label_info_dict.items():
                # 打开数据集图片
                src_image = cv2.imread(os.path.join(src_image_root_path, image_name))
                print(image_name)
                # 遍历每张图片中的boxs
                for text_info in text_info_list:
                    try:
                        text = text_info['label']
                        for char in text:
                            char_set.add(char)
                        src_point_list = text_info['points']
                        # ========================== 外接矩形+仿射透视 =======================
                        crop_image,img_size = Image_out(src_image, src_point_list)
                        # ====================== 如果竖直方向文本,右转90度 ====================
                        if 2*img_size[0]<img_size[1]:
                            print(global_image_num)
                            crop_image = np.rot90(crop_image, -1)
                        # =================================================================
                        if crop_image.size == 0:
                            continue
                        crop_image_name = '{}.jpg'.format(global_image_num)
                        global_image_num += 1
                        cv2.imwrite(os.path.join(save_image_path, crop_image_name), crop_image)
                        out_file.write('{}\t{}\n'.format(crop_image_name, text))
                        # 每次循环得到的增强图像保存
                        crop_image_name = '{}.jpg'.format(global_image_num)
                        global_image_num += 1
                        cv2.imwrite(os.path.join(save_image_path, crop_image_name), img_return)
                        out_file.write('{}\t{}\n'.format(crop_image_name, text))
                    except:
                        pass

        for image_name, text_info_list in label_info_dict.items():
            for text_info in text_info_list:
                text = text_info['label']
                text = text.replace('\r', '').replace('\n', '')
                for char in text:
                    char_set.add(char)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_train_image_path', type=str,
                        default='/path/to/tmp_data_test/recognizer_images')

    parser.add_argument('--save_train_txt_path', type=str,
                        default='/path/to/tmp_data_test/recognizer_txts')
    # test测试集图片路径
    parser.add_argument('--train_image_common_root_path', type=str,
                        default='/path/to/official_data/test_image')
    # test测试集.json文件路径
    parser.add_argument('--common_label_json_file', type=str,
                        default='/path/to/output/test_null.json')

    opt = parser.parse_args()

    # 文本训练集图片路径
    save_train_image_path = opt.save_train_image_path
    # 文本训练集标签路径
    save_train_txt_path = opt.save_train_txt_path

    # # common训练集
    train_image_common_root_path = opt.train_image_common_root_path
    common_label_json_file = opt.common_label_json_file
    extract_train_data(train_image_common_root_path,common_label_json_file,save_train_image_path,save_train_txt_path,enhance_num=0)


    print('Image num is {}.'.format(global_image_num))


