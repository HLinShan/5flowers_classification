# -*- coding = utf-8 -*-
# coding=utf-8

from __future__ import absolute_import,division,print_function

import numpy as np
import tensorflow as tf
import time
from scipy.misc import imread,imresize
from os import  walk
import os
from skimage import io,transform
from os.path import join
import glob
from sys import path

#图片存放位置
DATA_DIR = '/home/jack/sansan_progress/5flowers/flowers'

#图片信息
h = 150
w = 150
IMG_CHANNELS = 3
ratio=0.3



#读取图片
#读取图片
def read_images(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
data,label=read_images(DATA_DIR)
# shuffle
#打乱顺序
num_example=data.shape[0]
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]



#生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#生成字符串型的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert(images,labels,name):
    #获取要转换为TFRecord文件的图片数目
    num = images.shape[0]
    #输出TFRecord文件的文件名
    filename = name+'.tfrecords'
    print('Writting',filename)
    #创建一个writer来写TFRecord文件
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(num):
        #将图像矩阵转化为一个字符串
        img_raw = images[i].tostring()
        #将一个样例转化为Example Protocol Buffer，并将所有需要的信息写入数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(int(labels[i])),
            'image_raw': _bytes_feature(img_raw)}))
        #将example写入TFRecord文件
        writer.write(example.SerializeToString())
    writer.close()
    print('Writting End')

def main(argv):
    print('reading images begin')
    start_time = time.time()
    train_images,train_labels = read_images(DATA_DIR)
    print (train_images.shape[0])
    duration = time.time() - start_time
    print("reading images end , cost %d sec" %duration )

    #get validation
    num_example=train_images.shape[0]
    # s=num_example*ratio
    # validation_images = train_images[:s,:,:,:]
    # validation_labels = train_labels[:s]
    # # get train
    # train_images = train_images[s:,:,:,:]
    # train_labels = train_labels[s:]
    #
    # #convert to tfrecords
    # print('convert to tfrecords begin')
    # start_time = time.time()
    # convert(train_images,train_labels,'train')
    # convert(validation_images,validation_labels,'validation')
    # duration = time.time() - start_time
    # print('convert to tfrecords end , cost %d sec' %duration)

if __name__ == '__main__':
    tf.app.run()
