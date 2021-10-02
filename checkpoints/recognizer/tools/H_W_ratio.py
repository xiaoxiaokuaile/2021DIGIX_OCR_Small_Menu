# Learner: 王振强
# Learn Time: 2021/9/12 20:07
# -*- coding=utf-8 -*-
import argparse
import cv2
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 图片所在文件目录
    image_path = '/path/to/tmp_data_special/recognizer_images'
    # 保存切片名与labe的文档
    parser.add_argument('--src_train_file_path', type=str,default='/path/to/tmp_data_special/recognizer_txts/train.txt')

    opt = parser.parse_args()
    src_train_file_path = opt.src_train_file_path

    with open(src_train_file_path, 'r', encoding='utf-8') as in_file:
        lines = in_file.readlines()
        # 统计宽高比小于17的图片数目
        sum_1 = 0
        # 统计1.5倍以内
        sum_15 = 0
        # 统计宽高比大于17-35的图片数目
        sum_2 = 0
        # 统计宽高比大于35-53的数目
        sum_3 = 0
        # 统计宽高比大于53的数目
        sum_4 = 0



        for index, line in enumerate(lines):
            if index%10000 == 0:
                print(index)
            line = line.strip('\r').strip('\n')
            # line_list[0]代表图片名  line_list[1]代表label
            line_list = line.split('\t')
            if '#' in line_list[1]:
                continue
            if line_list[0].split('.')[1] != 'jpg':
                print(index, line)
            if len(line_list[-1]) <= 0:
                continue
            # 读取图片
            src_image = cv2.imread(os.path.join(image_path, line_list[0]))
            H = src_image.shape[0]
            W = src_image.shape[1]
            if W/H < 17.5:
                sum_1 = sum_1 + 1
            elif W/H < 26:
                sum_15 = sum_15 + 1
            elif W/H < 35:
                sum_2 = sum_2 + 1
            elif W/H < 53:
                sum_3 = sum_3 + 1
            else:
                sum_4 = sum_4 + 1


        print(sum_1,sum_15,sum_2,sum_3,sum_4)


'''bash
python recognizer/tools/from_text_to_label.py
'''