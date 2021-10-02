# -*- coding=utf-8 -*-
import argparse
import os
import sys

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(basedir)

import tensorflow as tf


from recognizer.models.crnn_model import crnn_model_based_on_densenet_crnn_time_softmax_activate
from recognizer.tools.config import config
from recognizer.tools.generator import Generator

# 或import tensorflow.keras.backend as K
K = tf.keras.backend

# ======================= 自定义学习策略所需 =============================
from tensorflow.keras.callbacks import LearningRateScheduler
def scheduler(epoch):
    # warmup
    if epoch < 1:
        K.set_value(model.optimizer.lr, 0.0003)
    elif epoch < 2:
        K.set_value(model.optimizer.lr, 0.0005)
    elif epoch < 3:
        K.set_value(model.optimizer.lr, 0.0007)
    # 每隔15个epoch，学习率减小为原来的1/10
    elif epoch % 10 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    print('lr',K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_dir',default='/path/to/menu_data/checkpoints/recognizer')
    parser.add_argument('--log_dir',default='/path/to/menu_data/output/recognizer_log')
    # 设置训练数据集
    parser.add_argument('--image_dir',default='/path/to/menu_data/tmp_data/recognizer_images')
    parser.add_argument('--txt_dir',default='/path/to/menu_data/tmp_data/recognizer_txts')
    # # 设置验证数据集
    # parser.add_argument('--val_image_dir',default='/path/to/menu_data/tmp_data_special/recognizer_images')
    # parser.add_argument('--val_txt_dir',default='/path/to/menu_data/tmp_data_special/recognizer_txts')
    parser = parser.parse_args()


    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.TF_ALLOW_GROWTH
    sess = tf.Session(config=sess_config)
    K.set_session(sess)

    batch_size = 64
    max_label_length = 239
    epochs = 35
    base_model, model = crnn_model_based_on_densenet_crnn_time_softmax_activate(initial_learning_rate=0.0007)

    # 训练数据集
    train_loader = Generator(root_path=parser.image_dir,
                             input_map_file=os.path.join(parser.txt_dir, 'real_train.txt'),
                             batch_size=batch_size,
                             max_label_length=max_label_length,
                             input_shape=(32, 560, 3),
                             is_training=True,
                             is_enhance=True)

    # 验证集数据
    # val_loader = Generator(root_path=parser.val_image_dir,
    #                          input_map_file=os.path.join(parser.val_txt_dir, 'real_train.txt'),
    #                          batch_size=batch_size,
    #                          max_label_length=max_label_length,
    #                          input_shape=(32, 560, 3),
    #                          is_training=False,
    #                          is_enhance=False)


    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(parser.model_save_dir, 'CNN_CTC_base_modle.h5'),
        # filepath=os.path.join(parser.model_save_dir, 'weights_crnn-{epoch:03d}-{loss:.3f}.h5'),
        monitor='loss', save_best_only=True, save_weights_only=True
    )

    # 调整学习率
    change_learning_rate = LearningRateScheduler(scheduler)

    # 训练模型
    model.fit_generator(generator=train_loader.__next__(),
                        steps_per_epoch=train_loader.num_samples() // batch_size,
                        epochs=epochs,
                        # validation_data=val_loader.__next__(),
                        # validation_steps=val_loader.num_samples() // batch_size,
                        initial_epoch=0,
                        workers = 4,
                        verbose=1,
                        use_multiprocessing=True,
                        callbacks=[checkpoint, change_learning_rate])


'''bash
    python recognizer/train.py
'''
