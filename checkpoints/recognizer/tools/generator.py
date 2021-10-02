# -*- coding=utf-8 -*-


import os
import random
import traceback
import time

import cv2
import numpy as np

from recognizer.tools.config import config

import imutils
from recognizer.tools.warp_mls import WarpMLS

class BatchIndices:
    def __init__(self, total_num, batch_size, is_training=True):
        self.total_num = total_num
        self.batch_size = batch_size
        self.is_training = is_training
        self.index = None
        self.curr = None
        self.is_epoch_end = False
        self.reset()

    def reset(self):
        self.index = np.random.permutation(self.total_num) if self.is_training else np.arange(0, self.total_num)
        self.curr = 0

    def __next__(self):
        self.is_epoch_end = False
        if self.curr >= self.total_num:
            self.reset()
            self.is_epoch_end = True
        remaining_next = min(self.batch_size, self.total_num - self.curr)
        res = self.index[self.curr:self.curr + remaining_next]
        self.curr += remaining_next
        return res, self.is_epoch_end


class Generator:
    def __init__(self, root_path, input_map_file, batch_size, max_label_length, input_shape, is_training,is_enhance):
        self.root_path = root_path
        self.input_map_file = input_map_file
        self.batch_size = batch_size
        self.max_label_length = max_label_length
        self.input_shape = input_shape
        self.is_training = is_training
        self.is_enhance = is_enhance
        self.epoch_time = 0
        self.image_to_label = self.parse_map_file()
        self.batch_indexes = BatchIndices(len(self.image_to_label), self.batch_size, self.is_training)

    def parse_map_file(self):
        res = list()
        with open(self.input_map_file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for i in lines:
                res.append(i.strip().replace('\n', '').replace('\r', ''))
        dic = dict()
        for i in res:
            path, values = i.split('\t')
            dic[path] = values.split(' ')
        return dic

    def num_samples(self):
        return len(self.image_to_label)

    # ========================================
    # 1.随机旋转
    def apply_rotate(self,img):
        rot = random.uniform(-5, 5)
        rotated = imutils.rotate_bound(img, rot)
        return rotated

    # 2.RGB空间做颜色随机扰动
    def PCA_Jittering(self,img):
        img = np.asanyarray(img, dtype='float32')
        img = img / 255.0
        img_size = img.size // 3  # 转换为单通道
        img1 = img.reshape(img_size, 3)
        img1 = np.transpose(img1)  # 转置
        img_cov = np.cov([img1[0], img1[1], img1[2]])  # 协方差矩阵
        lamda, p = np.linalg.eig(img_cov)  # 得到上述协方差矩阵的特征向量和特征值
        # p是协方差矩阵的特征向量
        p = np.transpose(p)  # 转置回去
        # 生成高斯随机数********可以修改
        alpha1 = random.gauss(0, 2)
        alpha2 = random.gauss(0, 2)
        alpha3 = random.gauss(0, 2)
        # lamda是协方差矩阵的特征值
        v = np.transpose((alpha1 * lamda[0], alpha2 * lamda[1], alpha3 * lamda[2]))  # 转置
        # 得到主成分
        add_num = np.dot(p, v)
        # 在原图像的基础上加上主成分
        img2 = np.array([img[:, :, 0] + add_num[0], img[:, :, 1] + add_num[1], img[:, :, 2] + add_num[2]])
        # 现在是BGR，要转成RBG再进行保存
        img2 = np.swapaxes(img2, 0, 2)
        img2 = np.swapaxes(img2, 0, 1)
        img2 = img2 * 255.0
        img2 = img2.astype(np.uint8)
        return img2

    # 3.gauss模糊
    def apply_gauss_blur(self,img, ks=None):
        if ks is None:
            ks = [3, 5]
        ksize = random.choice(ks)

        sigmas = [0, 1, 2, 3, 4, 5, 6, 7]
        sigma = 0
        if ksize >= 3:
            sigma = random.choice(sigmas)
        img = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        return img

    # 4.norm模糊
    def apply_norm_blur(self,img, ks=None):
        # kernel == 1, the output image will be the same
        if ks is None:
            ks = [2, 3]
        kernel = random.choice(ks)
        img = cv2.blur(img, (kernel, kernel))
        return img

    # 5.锐化
    def apply_emboss(self,img):
        emboss_kernal = np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ])
        return cv2.filter2D(img, -1, emboss_kernal)

    # 6.滤波,字体更细
    def apply_sharp(self,img):
        sharp_kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])
        return cv2.filter2D(img, -1, sharp_kernel)

    # 7.噪点增加
    def add_noise(self,img):
        for i in range(20):  # 添加点噪声
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x][temp_y] = 255
        return img

    # 8.扭曲
    def distort(self,src, segment=4):
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
    def stretch(self,src, segment=4):
        img_h, img_w = src.shape[:2]

        cut = img_w // segment
        thresh = cut * 4 // 5
        # thresh = img_h // segment // 3
        # thresh = img_h // 5

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
    def perspective(self,src):
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
    # ========================================

    def __next__(self):
        images_name = [image_name for image_name, image_label in self.image_to_label.items()]
        image_name_array = np.array(images_name)
        input_height, input_width, input_channel = self.input_shape
        sequence_length = 280
        start = 0
        while True:
            if config.is_debug:
                start = time.time()
            batch_index, is_epoch_end = next(self.batch_indexes)
            curr_bath_size = len(batch_index)
            try:
                batch_image_name_array = image_name_array[batch_index]
                label = np.ones([curr_bath_size, self.max_label_length]) * 10000
                input_length = np.zeros([curr_bath_size, 1])
                label_length = np.zeros([curr_bath_size, 1])
                input_images = np.zeros((curr_bath_size, input_height, input_width, input_channel), dtype=np.float)
                index = 0
                for image_name in batch_image_name_array:
                    try:
                        if input_channel == 1:
                            image = cv2.imread(os.path.join(self.root_path, image_name), cv2.IMREAD_GRAYSCALE)
                        else:
                            image = cv2.imread(os.path.join(self.root_path, image_name), cv2.IMREAD_COLOR)

                        # =====================================
                        # image = crop_image.copy()
                        if self.is_enhance:
                            # 0.7的概率进行数据增强
                            if random.random()<0.7:
                                mode = random.randint(0,9)
                                if mode == 0:
                                    image = self.apply_rotate(image)
                                elif mode == 1:
                                    image = self.PCA_Jittering(image)
                                elif mode == 2:
                                    image = self.apply_gauss_blur(image)
                                elif mode == 3:
                                    image = self.apply_norm_blur(image)
                                elif mode == 4:
                                    image = self.apply_emboss(image)
                                elif mode == 5:
                                    image = self.apply_sharp(image)
                                elif mode == 6:
                                    image = self.add_noise(image)
                                elif mode == 7:
                                    image = self.distort(image)
                                elif mode == 8:
                                    image = self.stretch(image)
                                elif mode == 9:
                                    image = self.perspective(image)
                                else:
                                    pass
                            else:
                                pass
                        # =====================================

                        scale = image.shape[0] * 1.0 / input_height
                        image_width = int(image.shape[1] // scale)
                        image = cv2.resize(image, (image_width, input_height))
                        image_height, image_width = image.shape[0:2]
                        if image_width <= input_width:
                            new_image = np.ones((input_height, input_width, input_channel), dtype='uint8')
                            new_image[:] = 255
                            if input_channel == 1:
                                image = np.expand_dims(image, axis=2)
                            new_image[:, :image_width, :] = image
                            image = new_image
                        else:
                            image = cv2.resize(image, (input_width, input_height))

                            if input_channel == 1:
                                image = np.expand_dims(image, axis=2)
                        image = np.array(image, 'f') / 127.5 - 1.0
                    except Exception as e:
                        print('skipped image {}. exception: {}'.format(image_name, e))
                        continue
                    input_images[index] = image
                    label_length[index] = len(self.image_to_label[image_name])
                    input_length[index] = sequence_length
                    label[index, :len(self.image_to_label[image_name])] = [int(k) for k in
                                                                           self.image_to_label[image_name]]
                    index += 1
                label = np.delete(label, [i for i in range(index, curr_bath_size)], axis=0)
                input_length = np.delete(input_length, [i for i in range(index, curr_bath_size)], axis=0)
                label_length = np.delete(label_length, [i for i in range(index, curr_bath_size)], axis=0)
                input_images = np.delete(input_images, [i for i in range(index, curr_bath_size)], axis=0)

                inputs = {'input_data': input_images,
                          'label': label,
                          'input_length': input_length,
                          'label_length': label_length,
                          }
                outputs = {'ctc': np.zeros([index])}
                self.epoch_time += (time.time() - start)
                if config.is_debug and is_epoch_end:
                    print("\nThe current total time for epoch to load data is {0}.".format(self.epoch_time))
                    self.epoch_time = 0
                del image, label, input_images, label_length, input_length
                yield inputs, outputs
            except Exception as e:
                print('{0} is wrong, error is {1}. {2}'.format(image_name_array[batch_index], str(e),
                                                               traceback.format_exc()))
                self.__next__()

