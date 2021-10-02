# -*- coding=utf-8 -*-
import argparse
import os
import json
import numpy as np
import cv2
import math
import random
import imutils
from warp_mls import WarpMLS

global_image_num = 0
char_set = set()

# 1.pply_rotate
def apply_rotate(img):
    rot = random.uniform(-5,5)
    rotated = imutils.rotate_bound(img, rot)
    return rotated

# 2.PCA_Jittering
def PCA_Jittering(img):
    img = np.asanyarray(img, dtype='float32')
    img = img / 255.0
    img_size = img.size // 3  
    img1 = img.reshape(img_size, 3)
    img1 = np.transpose(img1)  
    img_cov = np.cov([img1[0], img1[1], img1[2]])  
    lamda, p = np.linalg.eig(img_cov)  
    p = np.transpose(p)  
    alpha1 = random.gauss(0, 2)
    alpha2 = random.gauss(0, 2)
    alpha3 = random.gauss(0, 2)
    v = np.transpose((alpha1 * lamda[0], alpha2 * lamda[1], alpha3 * lamda[2])) 
    add_num = np.dot(p, v)
    img2 = np.array([img[:, :, 0] + add_num[0], img[:, :, 1] + add_num[1], img[:, :, 2] + add_num[2]])
    img2 = np.swapaxes(img2, 0, 2)
    img2 = np.swapaxes(img2, 0, 1)
    img2 = img2*255.0
    img2 = img2.astype(np.uint8)
    return img2

# 3.apply_gauss_blur
def apply_gauss_blur(img, ks=None):
    if ks is None:
        ks = [3,5]
    ksize = random.choice(ks)

    sigmas = [0, 1, 2, 3, 4, 5, 6, 7]
    sigma = 0
    if ksize >= 3:
        sigma = random.choice(sigmas)
    img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return img

# 4.apply_norm_blur
def apply_norm_blur(img, ks=None):
    # kernel == 1, the output image will be the same
    if ks is None:
        ks = [2, 3]
    kernel = random.choice(ks)
    img = cv2.blur(img, (kernel, kernel))
    return img

# 5.apply_emboss
def apply_emboss(img):
    emboss_kernal = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ])
    return cv2.filter2D(img, -1, emboss_kernal)

# 6.apply_sharp
def apply_sharp(img):
    sharp_kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    return cv2.filter2D(img, -1, sharp_kernel)

# 7.add_noise
def add_noise(img):
    for i in range(20): #添加点噪声
        temp_x = np.random.randint(0,img.shape[0])
        temp_y = np.random.randint(0,img.shape[1])
        img[temp_x][temp_y] = 255
    return img

# 8.扭曲
def distort(src, segment=4):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut // 3
    # thresh = img_h // segment // 3
    # thresh = img_h // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append([img_w - np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append([img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
    dst_pts.append([np.random.randint(thresh), img_h - np.random.randint(thresh)])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                        np.random.randint(thresh) - half_thresh])
        dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                        img_h + np.random.randint(thresh) - half_thresh])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst

# 9.伸展
def stretch(src, segment=4):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut * 4 // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, 0])
    dst_pts.append([img_w, 0])
    dst_pts.append([img_w, img_h])
    dst_pts.append([0, img_h])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        move = np.random.randint(thresh) - half_thresh
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + move, 0])
        dst_pts.append([cut * cut_idx + move, img_h])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst

# 10.透镜
def perspective(src):
    img_h, img_w = src.shape[:2]

    thresh = img_h // 2

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, np.random.randint(thresh)])
    dst_pts.append([img_w, np.random.randint(thresh)])
    dst_pts.append([img_w, img_h - np.random.randint(thresh)])
    dst_pts.append([0, img_h - np.random.randint(thresh)])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst

# =============================== two point's distance ==================================
def getDist_P2P(Point0, PointA):
    distance = math.pow((Point0[0] - PointA[0]), 2) + math.pow((Point0[1] - PointA[1]), 2)
    distance = math.sqrt(distance)
    return distance
# ========================== transformation parameter matrix ============================
def cal_perspective_params(img, points):
    img_size = (np.int32(getDist_P2P(points[0],points[1])),np.int32(getDist_P2P(points[0],points[2])))
    src = np.float32(points)
    dst = np.float32([[0, 0], [img_size[0], 0],[0, img_size[1]], [img_size[0], img_size[1]]])
    M = cv2.getPerspectiveTransform(src, dst)
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    return M, M_inverse,img_size

# ==================================== out picture =======================================
def Image_out(src_image,src_point_list):
    src_point_list = np.array(src_point_list)
    _min_area_box = cv2.minAreaRect(src_point_list.astype(np.float32))
    _min_area_box = np.array(cv2.boxPoints(_min_area_box), dtype=np.int32)
    sorted_point_list_by_x = sorted(_min_area_box, key=lambda x: x[0])
    left_points = sorted_point_list_by_x[:2]
    left_points_sort = sorted(left_points, key=lambda x: x[1])
    right_points = sorted_point_list_by_x[2:]
    right_points_sort = sorted(right_points, key=lambda x: x[1])
    points = [left_points_sort[0], right_points_sort[0], left_points_sort[1], right_points_sort[1]]
    M, M_inverse, img_size = cal_perspective_params(src_image, points)
    crop_image = cv2.warpPerspective(src_image, M, img_size)
    return crop_image,img_size

def extract_train_data(src_image_root_path, src_label_json_file, save_image_path, save_txt_path,enhance_num=2):
    global global_image_num, char_set
    with open(src_label_json_file, 'r', encoding='utf-8') as in_file:
        label_info_dict = json.load(in_file)
        with open(os.path.join(save_txt_path, 'train.txt'), 'a', encoding='utf-8') as out_file:
            for image_name, text_info_list in label_info_dict.items():
                # open image
                src_image = cv2.imread(os.path.join(src_image_root_path, image_name))
                print(image_name)
                for text_info in text_info_list:
                    try:
                        text = text_info['label']
                        for char in text:
                            char_set.add(char)
                        src_point_list = text_info['points']                   
                        # ==================== change =====================
                        crop_image,img_size = Image_out(src_image, src_point_list)
                        # ===================== rot90 =====================
                        if 2*img_size[0]<img_size[1]:
                            print(global_image_num)
                            crop_image = np.rot90(crop_image, -1)
                        # ==================================================
                        
                        if crop_image.size == 0:
                            continue
                        crop_image_name = '{}.jpg'.format(global_image_num)
                        global_image_num += 1
                        cv2.imwrite(os.path.join(save_image_path, crop_image_name), crop_image)
                        out_file.write('{}\t{}\n'.format(crop_image_name, text))

                        # Data enhancement
                        image = crop_image.copy()
                        if enhance_num>=10:
                            modes = [0,1,2,3,4,5,6,7,8,9]
                        else:
                            modes = random.sample(range(0, 10), enhance_num)
                        for mode in modes:
                            if mode == 0:
                                img_return = apply_rotate(image)
                            elif mode == 1:
                                img_return = PCA_Jittering(image)
                            elif mode == 2:
                                img_return = apply_gauss_blur(image)
                            elif mode == 3:
                                img_return = apply_norm_blur(image)
                            elif mode == 4:
                                img_return = apply_emboss(image)
                            elif mode == 5:
                                img_return = apply_sharp(image)
                            elif mode == 6:
                                img_return = add_noise(image)
                            elif mode == 7:
                                img_return = distort(image)
                            elif mode == 8:
                                img_return = stretch(image)
                            elif mode == 9:
                                img_return = perspective(image)
                            # save image and txt
                            crop_image_name = '{}.jpg'.format(global_image_num)
                            global_image_num += 1
                            cv2.imwrite(os.path.join(save_image_path, crop_image_name), img_return)
                            out_file.write('{}\t{}\n'.format(crop_image_name, text))

                    except:
                        print('error')
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
                        default='/path/to/menu_data/tmp_data/recognizer_images')
    parser.add_argument('--save_train_txt_path', type=str,
                        default='/path/to/menu_data/tmp_data/recognizer_txts')
    parser.add_argument('--train_image_common_root_path', type=str,
                        default='/path/to/menu_data/official_data/train_image_common')
    parser.add_argument('--common_label_json_file', type=str,
                        default='/path/to/menu_data/official_data/train_label_common.json')
    parser.add_argument('--train_image_special_root_path', type=str,
                        default='/path/to/menu_data/official_data/train_image_special')
    parser.add_argument('--special_label_json_file', type=str,
                        default='/path/to/menu_data/official_data/train_label_special.json')

    opt = parser.parse_args()

    save_train_image_path = opt.save_train_image_path
    save_train_txt_path = opt.save_train_txt_path

    train_image_common_root_path = opt.train_image_common_root_path
    common_label_json_file = opt.common_label_json_file
    # enhance_num 表示离线数据增强数目
    extract_train_data(train_image_common_root_path,
                       common_label_json_file,
                       save_train_image_path,
                       save_train_txt_path,
                       enhance_num=0)

    train_image_special_root_path = opt.train_image_special_root_path
    special_label_json_file = opt.special_label_json_file

    extract_train_data(train_image_special_root_path,
                       special_label_json_file,
                       save_train_image_path,
                       save_train_txt_path,
                       enhance_num=0)

    print('Image num is {}.'.format(global_image_num))

    char_list = list(char_set)
    char_list.sort()

    # chars.txt
    with open('chars.txt', 'a', encoding='utf-8') as out_file:
        for char in char_list:
            out_file.write('{}\n'.format(char))

'''bash
python recognizer/tools/extract_train_data.py
'''

