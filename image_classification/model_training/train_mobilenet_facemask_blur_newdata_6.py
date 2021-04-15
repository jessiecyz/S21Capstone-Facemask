import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import MobileNetV3_6c as MobileNetV3

import os
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片

import random


for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)


def img_box(img, bbox):
    [H, W, R] = np.shape(img)
    x_min = int((bbox[0]-bbox[2]/2)*W)
    x_max = int((bbox[0]+bbox[2]/2)*W)
    y_min = int((bbox[1]-bbox[3]/2)*H)
    y_max = int((bbox[1]+bbox[3]/2)*H)
    return img[y_min:y_max,x_min:x_max,0:3]

def data_load():

    file_exts = ('jpeg','png','jpg')

    images_list = []
    labels_list = []

    train_path1_image = './training_data/train/images'
    train_paths_image = [train_path1_image]

    train_path1_label = './training_data/train/labels'
    train_paths_label = [train_path1_label]


    for ind,path in enumerate(train_paths_image):
        for filename in os.listdir(path):
            if filename.endswith(file_exts):

                img_absolute_path = os.path.join(path,filename)
                img = mpimg.imread(img_absolute_path)

                (name,ext) = os.path.splitext(filename)

                anno_absolute_path = os.path.join(train_paths_label[ind],name+'.txt')
                if os.path.exists(anno_absolute_path):

                    fp = open(anno_absolute_path)
                    for line in fp.readlines():
                        data_str = line.split()
                        bbox = [float(i) for i in data_str]
                        label = int(bbox[0])
                        bbox = bbox[1:]

                        img_new=img_box(img,bbox)
                        if np.min(np.shape(img_new))<1:
                            continue
                        if img.max() <= 1.0:
                            resized = tf.image.resize_with_pad(img_new,64,64,)
                        else:
                            resized = tf.image.resize_with_pad(img_new/255.0,64,64,)

                        images_list.append(resized.numpy())
                        labels_list.append(np.array([label]))
                        

    train_images = np.zeros((len(images_list),64,64,3), dtype = float)
    train_labels = np.zeros((len(images_list),1), dtype = int)
    for ind,img in enumerate(images_list):
        train_images[ind] = img
        train_labels[ind] = labels_list[ind]


    images_list = []
    labels_list = []

    test_path1_image = './valid_data/valid/images'
    test_paths_image = [test_path1_image]

    test_path1_label = './valid_data/valid/labels'
    test_paths_label = [test_path1_label]


    for ind,path in enumerate(test_paths_image):
        for filename in os.listdir(path):
            if filename.endswith(file_exts):

                img_absolute_path = os.path.join(path,filename)
                img = mpimg.imread(img_absolute_path)

                (name,ext) = os.path.splitext(filename)

                anno_absolute_path = os.path.join(test_paths_label[ind],name+'.txt')
                if os.path.exists(anno_absolute_path):

                    fp = open(anno_absolute_path)
                    for line in fp.readlines():
                        data_str = line.split()
                        bbox = [float(i) for i in data_str]
                        label = int(bbox[0])
                        bbox = bbox[1:]

                        img_new=img_box(img,bbox)
                        if np.min(np.shape(img_new))<1:
                            continue
                        if img.max() <= 1.0:
                            resized = tf.image.resize_with_pad(img_new,64,64,)
                        else:
                            resized = tf.image.resize_with_pad(img_new/255.0,64,64,)

                        images_list.append(resized.numpy())
                        labels_list.append(np.array([label]))

    test_images = np.zeros((len(images_list),64,64,3), dtype = float)
    test_labels = np.zeros((len(images_list),1), dtype = int)
    for ind,img in enumerate(images_list):
        test_images[ind] = img
        test_labels[ind] = labels_list[ind]

    return (train_images,train_labels), (test_images, test_labels)

    
class DataLoader():
    def __init__(self):

        # dataset加载

        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = data_load()
        self.num_train, self.num_test = self.train_images.shape[0], self.test_images.shape[0]


    def get_batch_train(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_images)[0], batch_size)
        #need to resize images to input shape
        resized_images = tf.image.resize_with_pad(self.train_images[index],64,64,)
        return resized_images.numpy(), self.train_labels[index]

    def get_batch_test(self, batch_size):
        index = np.random.randint(0, np.shape(self.test_images)[0], batch_size)
        #need to resize images to input shape
        resized_images = tf.image.resize_with_pad(self.test_images[index],64,64,)
        return resized_images.numpy(), self.test_labels[index]


def train_mobilenet(batch_size, epoch):
    #dataLoader = DataLoader()
    (train_images,train_labels), (test_images, test_labels) = data_load()
    checkpoint = tf.keras.callbacks.ModelCheckpoint('{epoch}_6c_blur_weight.h5',
        save_weights_only=True,
        verbose=1,
        save_freq='epoch')

    # 详细参数见官方文档：https://tensorflow.google.cn/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator?hl=en
    data_generate = ImageDataGenerator(
        featurewise_center=False,# 将输入数据的均值设置为0
        samplewise_center=False, # 将每个样本的均值设置为0
        featurewise_std_normalization=False,  # 将输入除以数据标准差，逐特征进行
        samplewise_std_normalization=False,   # 将每个输出除以其标准差
        zca_epsilon=1e-6,        # ZCA白化的epsilon值，默认为1e-6
        zca_whitening=False,     # 是否应用ZCA白化
        rotation_range=10,        # 随机旋转的度数范围，输入为整数
        width_shift_range=0.1,   # 左右平移，输入为浮点数，大于1时输出为像素值
        height_shift_range=0.1,  # 上下平移，输入为浮点数，大于1时输出为像素值
        shear_range=0.,          # 剪切强度，输入为浮点数
        zoom_range=0.1,          # 随机缩放，输入为浮点数
        channel_shift_range=0.,  # 随机通道转换范围，输入为浮点数
        fill_mode='nearest',     # 输入边界以外点的填充方式，还有constant,reflect,wrap三种填充方式
        cval=0.,                 # 用于填充的值，当fill_mode='constant'时生效
        horizontal_flip=True,    # 随机水平翻转
        vertical_flip=False,     # 随机垂直翻转
        rescale=None,            # 重缩放因子，为None或0时不进行缩放
        preprocessing_function=None,  # 应用于每个输入的函数
        data_format='channels_last',   # 图像数据格式，默认为channels_last
        validation_split=0.0
      )

    net = MobileNetV3.build_mobilenet()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,decay=1e-6)
    net.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    net.fit(
        data_generate.flow(train_images, train_labels, 
            batch_size=batch_size, 
            shuffle=True, 
            #save_to_dir='resource/images'
        ), 
        steps_per_epoch=len(train_images) // batch_size,
        epochs=epoch,
        callbacks=[checkpoint],
        shuffle=True)

def test_mobilenet(model_path):
    #dataLoader = DataLoader()
    (train_images,train_labels), (test_images, test_labels) = data_load()
    net = MobileNetV3.build_mobilenet()
    net.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    net.build((1,64,64,3))
    net.load_weights(model_path)
   # pred_labels = net.predict_classes(test_images)
    net.evaluate(test_images, test_labels, verbose=2)


if __name__ == '__main__':
    #dataLoader = DataLoader()
    # 训练
    train_mobilenet(64, 140)
    # 测试
    test_mobilenet('./140_6c_blur_weight.h5') # Data augmentation + 80 epoch Adam lr 0.001 decay 1e-6 84.06; data 60 epoch Adam lr 0.001 decay 1e-7 65.69
