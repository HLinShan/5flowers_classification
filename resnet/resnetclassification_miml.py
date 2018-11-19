# -*- coding: utf-8 -*-

import os
import os.path as osp
import numpy as np
import pprint
from datetime import datetime
import tensorflow as tf
from ops import *
# ----
# from Accuracy import *
import scipy.io as scio
# from glob import glob
# import tensorlayer as tl
import time
# import matplotlib.pyplot as plt
from skimage import io

# from input_pipeline import inputs, test_inputs

# path = "/home/luoya/Desktop/Resnet/test2/"red_phase_128_180_test.tfrecords
# red_phase_128_180_test_fold4_test.tfrecords

flags = tf.flags
flags.DEFINE_string("tfrecord_filename_train", "./tfrecordsfile/train.tfrecords", "The name of dataset []")
flags.DEFINE_string("tfrecord_filename_test", "./tfrecordsfile/validation.tfrecords", "The name of dataset []")
flags.DEFINE_string("tfrecord_filename_val", "./data/tfrecords/cam_f5_val", "The name of dataset []")  # ckptnum
flags.DEFINE_integer('BATCH_SIZE', 4, 'The size of batch images [128]')
flags.DEFINE_integer('ckptnum', 100000, 'The size of batch images [128]')
flags.DEFINE_integer('iterations', 100000, 'The size of batch images [128]')
flags.DEFINE_float('LR', 0.0001, 'Learning rate of for Optimizer [3e-3]')
flags.DEFINE_integer('NUM_GPUS', 0, 'The number of GPU to use [1]')
flags.DEFINE_integer('CLASS', 5, 'The number of the output classes. [5]')
flags.DEFINE_integer("size", 150, "width of image  . [1]")
flags.DEFINE_integer("fil_num", 16, "width of image  . [1]")
flags.DEFINE_integer("fil_num_2", 64, "width of image  . [1]")

flags.DEFINE_integer("fil_num_d", 4, "width of image  . [1]")
flags.DEFINE_integer("repeat_B", 4, "width of image  . [1]")
flags.DEFINE_boolean('IS_TRAIN', False, 'True for train, else test. [True]')
flags.DEFINE_boolean('BN', True, 'True for train, else test. [True]')
flags.DEFINE_boolean('IS_CIFAR10', True, 'True for train, else test. [True]')
flags.DEFINE_boolean('L2', True, 'True for train, else test. [True]')
flags.DEFINE_boolean('LOAD_MODEL', True, 'True for load checkpoint and continue training. [True]')
# flags.DEFINE_string('MODEL_DIR', "%s"%(path),'If LOAD_MODEL, provide the MODEL_DIR. [./model/BEGAN/]')
flags.DEFINE_string('MODEL_DIR', './', 'If LOAD_MODEL, provide the MODEL_DIR. [./model/BEGAN/]')
flags.DEFINE_integer("sort", 5, "piece of sort out  . [1]")
flags.DEFINE_integer("output", 5, "the output of resnet  . [1]")
flags.DEFINE_string('model_path', "./models/", 'If LOAD_MODEL, provide the MODEL_DIR')
flags.DEFINE_boolean('IS_TEST_DATASET', True, 'True for test_data, else train_data. [True]')
FLAGS = flags.FLAGS

GPU_ID = FLAGS.NUM_GPUS  # get_gpu_id(gpu_num = FLAGS.NUM_GPUS)
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.25
config.gpu_options.allow_growth = False


def inference(images, batch_size=FLAGS.BATCH_SIZE, is_train=True, reuse=False, name='inference', BN=False):
    fil_num = FLAGS.fil_num
    frames = images
    with tf.variable_scope(name, reuse=reuse):

        name_ = "0_infer_block"
        x = LReLU(Conv2d(frames, output_dim=fil_num, name=name + name_, is_train=is_train, BN=BN), name=name + name_)
        print x.shape
        x = MaxPooling(x, name='MaxPooling_1')
        print x.shape
        for i in range(FLAGS.repeat_B - 1):
            print i
            if i == 1:
                name = "%s_infer_block_t1" % (i)
                x = infer_block(x, FLAGS.fil_num, name=name, is_train=is_train, BN=BN)
                x = Conv2d(x, output_dim=FLAGS.fil_num_2, k_h=1, k_w=1, strides=[1, 1, 1, 1], name='conv_1_1',
                           is_train=is_train, BN=BN)

                x = MaxPooling(x, name='MaxPooling_2')
                print x.shape
            if i == 2:
                name = "%s_infer_block_t1" % (i)
                x = infer_block(x, FLAGS.fil_num_2, name=name, is_train=is_train, BN=BN)
                print x.shape
            if i == 0:
                name = "%s_infer_block_t1" % (i)
                x = infer_block(x, FLAGS.fil_num, name=name, is_train=is_train, BN=BN)
                print x.shape

        name = "3_infer_block_t1"
        x = infer_block(x, FLAGS.fil_num_2, name=name, is_train=is_train, BN=BN)
        print x.shape

        x = Global_Average_Pooling(x, name='Global_Average_Pooling_1')
        print x.shape
        #        x = linear(tf.reshape(x,[FLAGS.BATCH_SIZE,-1]),64,name='linear_1')
        #        print x.shape
        x = linear(tf.reshape(x, [FLAGS.BATCH_SIZE, -1]), FLAGS.output, name='linear_2')
        softmax_x = tf.nn.softmax(x)
        print x.shape
        return x, softmax_x


def infer_block(x, fil_num, stride=1, name="", is_train=True, BN=False):
    name_ = 'Conv_BN_LReLu_0'
    result0 = LReLU(Conv2d(x, output_dim=fil_num, name=name + name_, is_train=is_train, BN=BN), name=name + name_)
    name_ = 'Conv_BN_LReLu_1'
    result1 = LReLU(Conv2d(result0, output_dim=fil_num, name=name + name_, is_train=is_train, BN=BN), name=name + name_)
    #    name_ =  'Conv_BN_ReLu_2'
    #    result2 =ReLU(   Conv2d(result1, output_dim = fil_num, name = name+name_, is_train = is_train, BN = BN)  , name = name +name_ )
    #    name_ =  'Conv_BN_ReLu_-1'
    #    restore =   Conv2d(result2, output_dim = 2, name = name+name_, is_train = is_train, BN = BN)
    return x + result1


# def concat_(x1, x2,axis_=3):
#    return tf.concat([x1, x2], axis = axis_)
#
def losses(logits, labels, num_classes=5, head=None):
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(logits - labels))
    return loss


def soft_losses(labels, logits):
    with tf.name_scope('loss'):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return loss


def sofmax_losses(labels, logits):
    with tf.name_scope('loss_sofmax'):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return loss


# def ssim_losses(logits, labels, batchsize = FLAGS.BATCH_SIZE, head=None):
#    with tf.name_scope('ssim_loss'):
#        ssim_loss = loss_ssim(logits, labels, batchsize = batchsize, c_dims = 3)
#    return ssim_loss

def train(BN=FLAGS.BN, is_train=True):
    GPU_ID = FLAGS.NUM_GPUS  # get_gpu_id(gpu_num = FLAGS.NUM_GPUS)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    sess = tf.Session()

    global_step = bias('global_step', [], trainable=False, bias_start=0)

    #    data_dir = './data/baseline/'
    #    (train_images, train_labels, test_images, test_labels) = inputs(data_dir, batch_size = FLAGS.BATCH_SIZE)

    #    print train_images, train_labels, test_images, test_labels
    #    asas
    #     read image_label
    train_images, train_labels = read_and_decode_image_label()

    #    train_labels = tf.transpose(train_labels, perm=[1,0])

    #    train_labels=tf.one_hot(train_labels,10,on_value=1,off_value=None,axis=1)
    # train_labels=tf.transpose(train_labels,perm=[1,0])
    train_labels = tf.cast(train_labels, tf.float32)
    print "train_labels"
    print train_labels.shape
    train_logits, prediction = inference(train_images, BN=BN, is_train=is_train)
    print "trian_logit"
    print train_logits.shape
    loss_1 = sofmax_losses(train_labels, train_logits)
    print "loss_shape"
    print loss_1.shape
    correct_prediction = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(train_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # learning rate  decay
    lr = tf.train.exponential_decay(
        learning_rate=FLAGS.LR,
        global_step=global_step,
        staircase=True,
        decay_steps=20000,
        decay_rate=0.1

    )
    var_list = tf.trainable_variables()
    print "total vars is : %s" % (cal_para(var_list))

    if FLAGS.L2 == True:
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.add_n(regularization_losses) + loss_1
        print 'using the L2 regualarization'
    #     optim
    optim = tf.train.AdamOptimizer(lr).minimize(
        loss,
        var_list=var_list,
        global_step=global_step
    )
    loss_train = tf.reduce_mean(loss)

    summary_op = tf.summary.merge([
        tf.summary.scalar('loss_train', loss_train),
        tf.summary.scalar('lr', lr),
        tf.summary.scalar('accuracy', accuracy)
    ])

    init = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver(max_to_keep=None)
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    #    test_images_, test_labels_ = sess.run([test_images, test_labels])
    #    print test_images_[:, :, :, 6:9].shape
    #    save_images_(test_images_[:, :, :, 6:9] , [2, 2],
    #                 './{}/_val_img{:04d}.png'.format(sample_dir, 1))
    #    asas

    start = 0
    if FLAGS.LOAD_MODEL:
        print(' [*] Reading checkpoints...')
        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = osp.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, osp.join(model_dir, ckpt_name))
            global_step = ckpt.model_checkpoint_path.split(
                '/')[-1].split('-')[-1]
            print('Loading success, global_step is %s' % global_step)
            start = int(global_step)
        else:
            print(' [*] Failed to find a checkpoint')
            start = 0

    print '******* start with %d *******' % start

    idx = start
    start_time = time.time()
    try:
        while not coord.should_stop():
            # print "start the sess"

            _, lr_val, summary_str, Accuracy_, train_loss_mean, prediction_, train_labels_ = sess.run(
                [optim, lr, summary_op, accuracy, loss_train, prediction, train_labels])

            idx += 1
            writer.add_summary(summary_str, idx)
            if ((idx % 100 == 0) or (idx == 1)):
                print "the parameter === begin "
                print "idx", idx
                print "lr_val", lr_val
                print "train_loss_mean", train_loss_mean
                print "addcuracy", Accuracy_
                print train_labels_
                print prediction_

            if idx % 5000 == 0:
                checkpoint_path = osp.join(model_dir, 'baseline_model.ckpt')
                saver.save(sess, checkpoint_path, global_step=idx)
                print('**********  baseline_model%d saved  **********' % idx)
            if idx == FLAGS.iterations + 1:
                coord.request_stop()
    except tf.errors.OutOfRangeError:
        checkpoint_path = osp.join(model_dir, 'baseline_model.ckpt')
        saver.save(sess, checkpoint_path, global_step=idx)
        print('**********  baseline_model%d saved  **********' % idx)
        print('Training done!')

    finally:
        coord.request_stop()
    print "stop"

    coord.join(threads)
    sess.close()


def evaluate_test(batch_size=64, size=FLAGS.size, c_dim=1, is_train=False, BN=FLAGS.BN):
    train_images, train_labels = read_and_decode_image_label()
    # num=train_images.shape[0]
    train_labels = tf.cast(train_labels, tf.float32)
    train_logits, prediction = inference(train_images, BN=BN, is_train=is_train)
    correct_prediction = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(train_labels, 1))
    #    print(correct_prediction.shape )
    #    asas
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    #    print accuracy
    init = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )
    sess = tf.Session(config=config)
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    saver = tf.train.Saver(max_to_keep=None)
    # writer = tf.summary.FileWriter(log_dir, sess.graph)

    Accuracy_all = 0
    pre_0_0_num = pre_1_1_num = pre_2_2_num = pre_3_3_num = 0.0
    print(' [*] Reading checkpoints...')
    model_dir = FLAGS.model_path
    ckpt = tf.train.get_checkpoint_state(model_dir)
    i = 0
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = osp.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, osp.join(model_dir, ckpt_name))
        print osp.join(model_dir, ckpt_name)
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Loading success, global_step is %s' % global_step)

    else:
        print(' [*] Failed to find a checkpoint')
    #    try:
    #        while not coord.should_stop():
    dataset_num = 500.0
    dataset_num_int = 500
    while (1):
        train_labels_, prediction_, Accuracy = sess.run([train_labels, prediction, accuracy])
        #           print  prediction_
        #           print type(train_labels_)
        i = i + 1
        print  i
        Accuracy_all = Accuracy_all + Accuracy
        '''
        pre_0_0_num_batch,pre_1_1_num_batch,pre_2_2_num_batch,pre_3_3_num_batch, p_3_0, p_3_1, p_3_2 = Accuracy_count(prediction_,train_labels_)  

#            if p_3_0!=0 or p_3_1!=0 or p_3_2!=0:
#            if 
#               print i 
#               print train_labels_
#               print prediction_
        Accuracy_all = Accuracy_all+Accuracy
        pre_0_0_num = pre_0_0_num + pre_0_0_num_batch
        pre_1_1_num = pre_1_1_num + pre_1_1_num_batch
        pre_2_2_num = pre_2_2_num + pre_2_2_num_batch
        pre_3_3_num = pre_3_3_num + pre_3_3_num_batch
#            if pre_0_0_num ==456 and pre_3_3_num_batch != 4:
#               print i 
#               print train_labels_
#               print prediction_
        i=i+1
        print 'i = ',i
        print pre_0_0_num
        print pre_1_1_num
        print pre_2_2_num
        print pre_3_3_num
        if FLAGS.IS_TEST_DATASET == True:
           dataset_num = 114.0
           dataset_num_int = 114
        if FLAGS.IS_TEST_DATASET == False:
           dataset_num = 500.0
           dataset_num_int = 500
        if i == dataset_num_int:
              Accuracy_result = Accuracy_all/dataset_num
              pre_0_0_num_result = pre_0_0_num/dataset_num
              pre_1_1_num_result = pre_1_1_num/dataset_num
              pre_2_2_num_result = pre_2_2_num/dataset_num
              pre_3_3_num_result = pre_3_3_num/dataset_num
              print pre_0_0_num+pre_1_1_num+pre_2_2_num+pre_3_3_num,dataset_num

              print 'Accuracy_result:'
              print Accuracy_result
              print 'accuracy_0_0'
              print pre_0_0_num_result
              print 'accuracy_1_1'
              print pre_1_1_num_result
              print 'accuracy_2_2'
              print pre_2_2_num_result
              print 'accuracy_3_3'
              print pre_3_3_num_result               
#                  coord.request_stop()
              break
    '''
        if i == dataset_num_int:
            Accuracy_result = Accuracy_all / dataset_num
            coord.request_stop()
            break
    #    except tf.errors.OutOfRangeError:
    #        print('Testing done!')

    #    finally:
    #        coord.request_stop()
    print "the total test accuracy is:"
    print Accuracy_result
    coord.join(threads)
    sess.close()


def read_and_decode_image_label():
    if FLAGS.IS_TRAIN == True:
        filename_queue = tf.train.string_input_producer([FLAGS.tfrecord_filename_train], shuffle=True)
    if FLAGS.IS_TRAIN == False:  ######num_epochs need to modify
        print 'test_dataset'
        filename_queue = tf.train.string_input_producer([FLAGS.tfrecord_filename_train], shuffle=False)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string),
            # 'row_img': tf.FixedLenFeature([], tf.int64),
            # 'col_img': tf.FixedLenFeature([], tf.int64),
        }
    )

    #    height = tf.cast(features['row_img'], tf.int32)
    #     img_row = tf.cast(features['row_img'], tf.int32)
    #     img_col = tf.cast(features['col_img'], tf.int32)
    img_label = tf.cast(features['label'], tf.int32)
    if img_label == 10:
        img_label = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    else:
        img_label = tf.one_hot(img_label, FLAGS.sort, on_value=1, off_value=None, axis=-1)

    img_raw = features['img_raw']
    img_raw = tf.decode_raw(img_raw, tf.float32)
    img_raw = tf.reshape(img_raw, [150, 150, 3])
    img_raw = tf.cast(img_raw, dtype=tf.float32)
    _time = time.time()

    if FLAGS.IS_CIFAR10 == False:
        img_raw = tf.random_crop(img_raw, [FLAGS.size, FLAGS.size, 3], seed=_time)
        print "cifar10"
    if FLAGS.IS_TRAIN == True:
        img_raw, img_label = tf.train.shuffle_batch([img_raw, img_label],
                                                    batch_size=FLAGS.BATCH_SIZE,
                                                    capacity=300,
                                                    min_after_dequeue=200)
    if FLAGS.IS_TRAIN == False:
        img_raw, img_label = tf.train.batch([img_raw, img_label],
                                            batch_size=FLAGS.BATCH_SIZE,
                                            capacity=300)

    print "////////////"
    print img_raw.shape
    print img_label.shape
    return img_raw, img_label


if __name__ == '__main__':

    if FLAGS.IS_TRAIN == True:
        log_dir = osp.join('logs', FLAGS.MODEL_DIR)
        model_dir = osp.join('models', FLAGS.MODEL_DIR)
        # test_dir = osp.join('test', FLAGS.MODEL_DIR)
        # sample_dir = osp.join('samples', FLAGS.MODEL_DIR)

        if not osp.exists(log_dir):
            os.makedirs(log_dir)
        if not osp.exists(model_dir):
            os.makedirs(model_dir)
        #    if not osp.exists(sample_dir):
        #        os.makedirs(sample_dir)
        #    if not osp.exists(test_dir):
        #        os.makedirs(test_dir)
        print('Current time: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print('The network initialization with learning rate %f ...' % FLAGS.LR)
        pprint.pprint(FLAGS.__flags)
        train()
    else:
        evaluate_test()
