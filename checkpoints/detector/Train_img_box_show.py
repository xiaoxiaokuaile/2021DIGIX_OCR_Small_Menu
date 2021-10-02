# Learner: 王振强
# Learn Time: 2021/8/20 19:15
import os
import json
import cv2
import numpy as np
# ==================== 将训练集数据框标注出来 ========================
class OCR_DatasetCatalog(object):
    # 输入训练集还是测试集标注，数据集路径/menu_data/official_data'
    def __init__(self, dataset='train',image_path='',label_path = ''):
        super(OCR_DatasetCatalog, self).__init__()
        self.image_path = image_path
        self.label_path = label_path
        if  dataset == 'train':
            self.general_datasets = {
                'common_train': {
                    'root_path': 'train_image_common/',
                    'gt_path': 'train_label_common.json',
                },
                'special_train':{
                  'root_path':'train_image_special/',
                  'gt_path':'train_label_special.json',
                }
            }
        elif dataset == 'test':
            self.general_datasets = {
                'special_test': {
                    'root_path': 'test_image/',
                    'gt_path': 'test_null.json',
                }
            }

    def get(self, name):
        if name in self.general_datasets.keys():
            return self.general_datasets[name]
        else:
            raise RuntimeError('Dataset not available: {}.'.format(name))

# 得到训练集每张图片绝对路径 + 每张图片标签列表
def _list_files(DATASET_NAME_LIST,ocr_dataset_catalog):
    img_paths = []
    labels = []
    # 普通训练集与特殊训练集
    for dataset_idx, dataset_name in enumerate(DATASET_NAME_LIST):
        # 输出1. menu_train:
        print('{dataset_idx}. {dataset_name}: '.format(dataset_idx=dataset_idx + 1, dataset_name=dataset_name), end='')
        # dataset_attrs返回两个字典'root_path': 'train_image_common/','gt_path': 'train_label_common.json',
        dataset_attrs = ocr_dataset_catalog.get(dataset_name)
        # 得到训练集图片路径
        image_path = os.path.join(ocr_dataset_catalog.image_path, dataset_attrs['root_path'])
        # 得到训练集标签路径
        train_json_path = os.path.join(ocr_dataset_catalog.label_path) + '/' + dataset_attrs['gt_path']
        # ======================= 改动 ===========================
        with open(train_json_path, 'r', encoding='utf-8') as f:
            gt = f.read()

        labels_dict = json.loads(gt)
        for image_name in labels_dict:
            # 拼接得到每张图片绝对路径:'train_image_common/'+ train_image_common_0.PNG
            img_paths.append(os.path.join(image_path, image_name))
            # 每张图片对应一个列表形式label [{"label": "030718", "points": [][][][]},{},{},...]
            labels.append(labels_dict[image_name])

    print('***sum_samples: {}'.format(len(img_paths)))
    return img_paths, labels

if __name__ == '__main__':
    # ------------ 输入要标定的数据集 -------------
    #MODE = 'train'
    MODE = 'test'
    # ------------------------------------------
    if MODE == 'train':
        DATASET_NAME_LIST = ['common_train', 'special_train']
        image_path = '/path/to/official_data'
        label_path = image_path
        # 保存路径
        dst_path = '/path/to/output/detector_train_box/'
    elif MODE == 'test':
        DATASET_NAME_LIST = ['special_test']
        image_path = '/path/to/official_data'
        label_path = '/path/to/output'
        # 保存路径
        dst_path = '/path/to/output/dtetector_test_boxs'

    DatasetClass = OCR_DatasetCatalog(MODE,image_path,label_path)
    img_paths, labels = _list_files(DATASET_NAME_LIST,DatasetClass)

    # 存放标注图片的文件夹
    if not os.path.isdir(dst_path):
        os.makedirs(dst_path)

    # 遍历训练集所有图片并绘制标注框
    for img_path,label in zip(img_paths, labels):
        ori_image = cv2.imread(img_path)
        # 获取该图片名称
        img_name = os.path.split(img_path)[-1]
        # 遍历每张图片中的box
        for box in label:
            box_num = box['points']
            # 绘制轮廓
            cv2.drawContours(ori_image, [np.array(box_num,dtype=np.int32)], 0, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(dst_path,'{}'.format(img_name)), ori_image)
        print(img_name)

