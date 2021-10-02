import os
import cv2
import math
import argparse
import operator
import collections
import sys
import torch
import json

import numpy as np

from tqdm import tqdm
from functools import reduce

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(basedir)

from detector.config import cfg
from detector.models import OCR_DETECTOR
from detector.dataset import get_test_dataset
from detector.postprocess import simple_dilate


decoder_methods = {'SIMPLE_DILATE': simple_dilate}

tt = {}


def predict(model, data_loader, dst_path, decoder_method, output_box):
    for batch_data in tqdm(data_loader):
        # get data
        ori_image, image, image_name, scale, ori_name = batch_data
        tt[ori_name[0]] = []
        ori_image = np.array(ori_image[0], dtype=np.uint8)
        image_name = image_name[0]
        # model.get_scale() = 4
        scale = 1.0 / scale[0].numpy() * (4.0 / model.get_scale())

        # model inference
        if os.environ['CUDA_VISIBLE_DEVICES'] is not None:
            if torch.cuda.is_available():
                image = image.cuda()

        # 从model出来得到两个1920×1952大小的tensor
        score, kernels = model(image)

        score = score.cpu().detach().numpy()
        kernels = kernels.cpu().detach().numpy()

        # decoder
        polygons = decoder_method(score, kernels)
        _Polyghons = []
        for polygon in polygons:
            _polygon = polygon.reshape(-1, 2) * scale
            _Polyghons.append(_polygon)

        # save result
        if not os.path.isdir(os.path.join(dst_path, 'menu/')):
            os.makedirs(os.path.join(dst_path, 'menu/'))

        # 输出外接矩形
        if output_box:
            _Min_area_box = []
            for polygon in _Polyghons:
                t = {}
                # 生成最小外接矩形，polygon为需要生成外接矩形的点集np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
                # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
                # =========== 可以从这里入手将偏斜的外接矩形改为平行于坐标轴的外接矩形 =============
                _min_area_box = cv2.minAreaRect(polygon.astype(np.float32))
                # 判定预测框类型
                if _min_area_box[1][0]/_min_area_box[1][1]<1.8 and _min_area_box[1][0]/_min_area_box[1][1]>0.6:
                    # =========================== 宽高比近似为1的字符不进行外接矩形 ==========================
                    sorted_point_list_by_x = sorted(polygon, key=lambda x: x[0])
                    sorted_point_list_by_y = sorted(polygon, key=lambda x: x[1])
                    # 左上角点
                    left_up_point = (sorted_point_list_by_x[0][0],sorted_point_list_by_y[0][1])
                    # 右上角点
                    right_up_point = (sorted_point_list_by_x[-1][0], sorted_point_list_by_y[0][1])
                    # 右下角点
                    right_down_point = (sorted_point_list_by_x[-1][0], sorted_point_list_by_y[-1][1])
                    # 左下角点
                    left_down_point = (sorted_point_list_by_x[0][0], sorted_point_list_by_y[-1][1])
                    # 得到字符轮廓
                    _min_area_box = np.array([left_up_point,right_up_point,right_down_point,left_down_point], dtype=np.int32)
                else:
                    # cv2.boxPoints(_min_area_box) 获取最小外接矩形4个顶点坐标
                    # 右下、左下、左上、右上
                    _min_area_box = np.array(cv2.boxPoints(_min_area_box), dtype=np.int32)
                # 绘制轮廓
                # cv2.drawContours(ori_image, [_min_area_box], 0, (0, 255, 0), 2)
                # 出现出界边框归零
                _min_area_box[_min_area_box < 0] = 0
                # _min_area_box.tolist() 将数组化为嵌套列表返回
                bb = _min_area_box.tolist()
                t["label"] = ""
                t["points"] = bb
                # print(bb)
                tt[ori_name[0]].append(t)
                _Min_area_box.append(_min_area_box)
            write_result_as_txt(_Min_area_box, os.path.join(dst_path, 'menu/{}.txt'.format(image_name)))
        else:
            for polygon in _Polyghons:
                cv2.drawContours(ori_image, [polygon.astype(np.int32)], 0, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(dst_path, 'menu/{}.JPEG'.format(image_name)), ori_image)

# 按顺序
def sort_to_clockwise(points):
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points), [len(points)] * 2))
    clockwise_points = sorted(points, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360, reverse=True)
    return clockwise_points

# 将每张预测图片的box写入单独的txt文档中
def write_result_as_txt(bboxes, dst_path):
    dir_path = os.path.split(dst_path)[0]
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    lines = []
    for bbox in bboxes:
        bbox = bbox.reshape(-1, 2)
        # 每个box平铺为一行8个数字
        bbox = np.array(list(sort_to_clockwise(bbox)))[[3, 0, 1, 2]].copy().reshape(-1)
        values = [int(v) for v in bbox]
        line = "%d,%d,%d,%d,%d,%d,%d,%d\n" % tuple(values)
        lines.append(line)

    with open(dst_path, 'w') as f:
        for line in lines:
            f.write(line)


if __name__ == '__main__':
    # get config
    parser = argparse.ArgumentParser('Hyperparams')
    parser.add_argument('--config', type=str, default=r'detector/config/resnet50.yaml', help='config yaml file.')
    parser.add_argument('--cuda', type=str, default="0")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    # 将args.config参数赋值给cfg
    cfg.merge_from_file(args.config)
    cfg.freeze()

    # model
    print('model info:')
    print('=> 1.backbone: ', cfg.MODEL.BACKBONE.ARCH)
    print('=> 2.neck: ', cfg.MODEL.NECK.ARCH)
    print('=> 3.head: ', cfg.MODEL.HEAD.ARCH)
    model = OCR_DETECTOR(cfg=cfg)
    if os.environ['CUDA_VISIBLE_DEVICES'] is not None:
        if torch.cuda.is_available():
            model = model.cuda()

    print("loading trained model '{}'".format(cfg.MODEL.TEST.CKPT_PATH))
    ckpt = torch.load(cfg.MODEL.TEST.CKPT_PATH,map_location=torch.device('cpu'))

    state_dict = ckpt if 'state_dict' not in ckpt.keys() else ckpt['state_dict']
    model_state_dict = collections.OrderedDict()

    for key, value in state_dict.items():
        if key.startswith('module'):
            _key = '.'.join(key.split('.')[1:])
        else:
            _key = key
        model_state_dict[_key] = value
    model.load_state_dict(model_state_dict)
    print('load trained parameters successfully.')
    model.eval()

    # 加载测试集
    data_loader = get_test_dataset(cfg)
    print('Next predict')

    with torch.no_grad():
        predict(model, data_loader,
                cfg.MODEL.TEST.RES_PATH,
                decoder_methods[cfg.MODEL.TEST.DECODER_METHOD],
                cfg.MODEL.TEST.OUTPUT_BOX
                )

    # 保存预测结果
    with open(os.path.join(cfg.MODEL.TEST.NULL_JSON_PATH, "test_null.json"), "w")as f:
        # json.dumps()将tt编码为json格式数据
        qq = json.dumps(tt)
        f.write(qq)


'''bash
    python detector/predict.py
'''



