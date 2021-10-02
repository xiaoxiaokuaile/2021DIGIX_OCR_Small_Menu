import os


class OCR_DatasetCatalog(object):
    # root_path:/menu_data/official_data
    def __init__(self, root_path=''):
        super(OCR_DatasetCatalog, self).__init__()
        self.root_path = root_path
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

    def get(self, name):
        if name in self.general_datasets.keys():
            return self.general_datasets[name]
        else:
            raise RuntimeError('Dataset not available: {}.'.format(name))


if __name__ == '__main__':
    ocr_dataset_catalog = OCR_DatasetCatalog()
    print(ocr_dataset_catalog.general_datasets.keys())
