import math
import numpy as np 
import tensorflow as tf
import tensorflow.contrib.layers as ly
import scipy.io as scio
import cv2

from tensorflow.python.framework import ops

from utils import *

#try:
#  image_summary = tf.image_summary
#  scalar_summary = tf.scalar_summary
#  histogram_summary = tf.histogram_summary
#  merge_summary = tf.merge_summary
#  SummaryWriter = tf.train.SummaryWriter
#except:
#  image_summary = tf.summary.image
#  scalar_summary = tf.summary.scalar
#  histogram_summary = tf.summary.histogram
#  merge_summary = tf.summary.merge
#  SummaryWriter = tf.summary.FileWriter
#
#if "concat_v2" in dir(tf):
#  def concat(tensors, axis, *args, **kwargs):
#    return tf.concat_v2(tensors, axis, *args, **kwargs)
#else:
#  def concat(tensors, axis, *args, **kwargs):
#    return tf.concat(tensors, axis, *args, **kwargs)
#
#class batch_norm(object):
#  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
#    with tf.variable_scope(name):
#      self.epsilon  = epsilon
#      self.momentum = momentum
#      self.name = name
#
#  def __call__(self, x, train=True):
#    return tf.contrib.layers.batch_norm(x,
#                      decay=self.momentum, 
#                      updates_collections=None,
#                      epsilon=self.epsilon,
#                      scale=True,
#                      is_training=train,
#                      scope=self.name)
#
#def conv_cond_concat(x, y):
#  """Concatenate conditioning vector on feature map axis."""
#  x_shapes = x.get_shape()
#  y_shapes = y.get_shape()
#  return concat([
#    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)
#
#def conv2d(input_, output_dim, 
#       k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
#       name="conv2d"):
#  with tf.variable_scope(name):
#    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
#              initializer=tf.truncated_normal_initializer(stddev=stddev))
#    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
#
#    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
#    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
#
#    return conv
#
#def deconv2d(input_, output_shape,
#       k_h=3, k_w=3, d_h=1, d_w=1, stddev=0.02,
#       name="deconv2d", with_w=False):
#  with tf.variable_scope(name):
#    # filter : [height, width, output_channels, in_channels]
#    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
#              initializer=tf.random_normal_initializer(stddev=stddev))
#    
#    try:
#      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
#                strides=[1, d_h, d_w, 1])
#
#    # Support for verisons of TensorFlow before 0.7.0
#    except AttributeError:
#      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
#                strides=[1, d_h, d_w, 1])
#
#    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
#    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
#
#    if with_w:
#      return deconv, w, biases
#    else:
#      return deconv
#     
#def lrelu(x, leak=0.2, name="lrelu"):
#  return tf.maximum(x, leak*x)
#
#def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
#  shape = input_.get_shape().as_list()
#
#  with tf.variable_scope(scope or "Linear"):
#    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
#                 tf.random_normal_initializer(stddev=stddev))
#    bias = tf.get_variable("bias", [output_size],
#      initializer=tf.constant_initializer(bias_start))
#    if with_w:
#      return tf.matmul(input_, matrix) + bias, matrix, bias
#    else:
#      return tf.matmul(input_, matrix) + bias

def cal_para(var_list):
    total_var = 0
    for var in var_list:
        shape = var.get_shape()
        var_p = 1
        for dim in shape:
            var_p *= dim.value
        total_var += var_p
    return total_var
    
def bias(name, shape, bias_start = 0.1, trainable = True):
    var = tf.get_variable(name, shape, tf.float32, trainable = trainable,
        initializer = tf.constant_initializer(bias_start, dtype = tf.float32))
    return var
    
def linear(value,output_size,stddev=0.02,bias_start=0.0,name="linear"):
    shape = value.get_shape().as_list()
    
    with tf.variable_scope(name):
        w = tf.get_variable("w",[shape[1],output_size],tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('b',[output_size],
                            initializer =tf.constant_initializer(bias_start)) 
        return tf.matmul(value,w) + b
                            
                            
                            
                            
                            
def Deconv2d(value, output_shape, k_h = 3, k_w = 3, strides =[1, 2, 2, 1],
             name = 'Deconv2d', with_w = False):
    with tf.variable_scope(name):
        weights = weight(name+'weights',
                         [k_h, k_w, output_shape[-1], value.get_shape()[-1]])
        deconv = tf.nn.conv2d_transpose(value, weights,
                                        output_shape, strides = strides)
        biases = bias(name+'biases', [output_shape[-1]])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, weights, biases
        else:
            return deconv


def Conv2d(value, output_dim, k_h = 3, k_w = 3,
           strides =[1, 1, 1, 1], name = 'Conv2d',is_train = True, BN= False, padding = 'SAME'):
    with tf.variable_scope(name):
        weights = weight(name+'weights',
                         [k_h, k_w, value.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(value, weights, strides = strides, padding = padding)
        biases = bias(name+'biases', [output_dim])
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        
        if BN:
            conv =BatchNorm(conv, is_train = is_train)
        else:
            conv = conv
        
        return conv
    
def MaxPooling(value, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
               padding = 'SAME', name = 'MaxPooling'):
    with tf.variable_scope(name):
        return tf.nn.max_pool(value, ksize = ksize,
                              strides = strides, padding = padding)
 
def AvgPooling(value, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
               padding = 'SAME', name = 'AvgPooling'):
    with tf.variable_scope(name):
        return tf.nn.avg_pool(value, ksize = ksize,
                              strides = strides, padding = padding)

def Global_Average_Pooling(value,strides =[1, 2, 2, 1],name ='Global_Average_Pooling'):
    height = value.shape[1]
    wideth = value.shape[2]
    with tf.variable_scope(name):
         return tf.nn.avg_pool(value,ksize=[1,height,wideth,1],strides = strides, padding='VALID')


def Concat(value, cond, name = 'concat'):
    """
    Concatenate conditioning vector on feature map axis.
    """

    with tf.variable_scope(name):
        value_shapes = value.get_shape().as_list()
        cond_shapes = cond.get_shape().as_list()
        return tf.concat([value,
              cond * tf.ones(value_shapes[0:3] + cond_shapes[3:])], axis = 3)


def BatchNorm(value, is_train = True, name = 'BatchNorm',
              epsilon = 1e-5, momentum = 0.9):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(
            value,
            decay = momentum,
            updates_collections = None,
            epsilon = epsilon,
            scale = True,
            is_training = is_train,
            scope = name
        )


#def GlobalAvePooling(value, name = 'GlobalAvePooling'):
#    with tf.variable_scope(name):
#        assert value.get_shape().ndims == 4
#        return tf.reduce_mean(value, [1, 2])

    
def ResizeNearNeighbor(value, scale = 2, name = 'Resize'):
    with tf.variable_scope(name):
        _, h, w, _ = value.get_shape().as_list()
        return tf.image.resize_nearest_neighbor(
            value, [h * scale, w * scale], name = name)
            
def weight(name, shape, stddev = 0.02, trainable = True):
    var = tf.get_variable(name, shape, tf.float32, trainable = trainable,
        initializer = tf.contrib.layers.xavier_initializer(dtype = tf.float32),
        regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4))
    return var

def save_images_(images, size, image_path, bound_ = True):
    images = images #* 255.0
#    images = (images *127.5)+127.5
    if bound_:
        images[np.where(images < 0 )] = 0.
        images[np.where(images > 255 )] = 255.
    else:
        pass
    images = images.astype(np.uint8)    
    return scipy.misc.imsave(image_path, merge(images, size))
    
def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB fuction
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1 : size//2 + 1]
    x_data = np.expand_dims(x_data, axis = -1)
    x_data = np.expand_dims(x_data, axis = -1)
    
    y_data = np.expand_dims(y_data, axis = -1)
    y_data = np.expand_dims(y_data, axis = -1)
    
    x = tf.constant(x_data, dtype = tf.float32)
    y = tf.constant(y_data, dtype = tf.float32)
    
    g = tf.exp(-((x ** 2 + y ** 2) / (2.0 * sigma * 2)))
    return g / tf.reduce_sum(g)
    
def tf_ssim(img1, img2, cs_map = False, mean_metric = True, size = 11, sigma = 1.5):
    window = _tf_fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 1
    C1 = (K1 * L) **2
    C2 = (K2 * L) **2
    mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
    mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = tf.nn.conv2d(img1 * img1, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1 * img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2
    if cs_map:
        value = (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
                  (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if mean_metric:
        value = tf.reduce_mean(value)
    return value
    
def image_to_4d(image):
    image = tf.expand_dims(image, 0)
    image = tf.expand_dims(image, -1)
    return image
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype = tf.float32))
    return numerator / denominator
    
def loss_ssim(img1, img2, batchsize = 4, c_dims = 3):
    ssim_value_sum = 0
    for i in range(batchsize):
        for j in range(c_dims):
            img1_tmp = img1[i,:,:,j]
            img1_tmp = image_to_4d(img1_tmp)
            img2_tmp = img2[i,:,:,j]
            img2_tmp = image_to_4d(img2_tmp)
            ssim_value_tmp = tf_ssim(img1_tmp, img2_tmp)
            ssim_value_sum += ssim_value_tmp
    ssim_value_ave = ssim_value_sum / (batchsize * c_dims)
    return log10(1.0 / (ssim_value_ave + 1e-4))
    
def loss_ssim_psnr(img1, img2, batchsize = 4, c_dims = 3):
    ssim_value_sum = 0
    for i in range(batchsize):
        for j in range(c_dims):
            img1_tmp = img1[i,:,:,j]
            img1_tmp = image_to_4d(img1_tmp)
            img2_tmp = img2[i,:,:,j]
            img2_tmp = image_to_4d(img2_tmp)
            ssim_value_tmp = tf_ssim(img1_tmp, img2_tmp)
            ssim_value_sum += ssim_value_tmp
    ssim_value_ave = ssim_value_sum / (batchsize * c_dims)
    loss_psnr = tf.reduce_mean(tf.square(tf.abs(img1 - img2)))
    loss_ssim = log10(1.0 / (ssim_value_ave + 1e-4))
    return (0.5 * loss_psnr + 0.5 * loss_ssim)
    
def loss_ssim_l1(img1, img2, batchsize = 4, c_dims = 3):
    ssim_value_sum = 0
    for i in range(batchsize):
        for j in range(c_dims):
            img1_tmp = img1[i,:,:,j]
            img1_tmp = image_to_4d(img1_tmp)
            img2_tmp = img2[i,:,:,j]
            img2_tmp = image_to_4d(img2_tmp)
            ssim_value_tmp = tf_ssim(img1_tmp, img2_tmp)
            ssim_value_sum += ssim_value_tmp
    ssim_value_ave = ssim_value_sum / (batchsize * c_dims)
    loss_l1 = tf.reduce_mean(tf.abs(img1 - img2))
    loss_ssim = log10(1.0 / (ssim_value_ave + 1e-4))
    return (0.5 * loss_l1 + 0.5 * loss_ssim)

def LReLU(x, leak = 0.2, name = 'LReLU'):
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x, name = name)   
    
#LP = scio.loadmat('./LP.mat')
#LP = LP['h']
#
#LP = np.reshape(LP,[5,5,1,1]).astype(np.float32)
#LP_ = LP
#for i in range(14):
#    LP = np.concatenate((LP, LP_), 2)


def hpass(x):
    LP_tf = tf.constant(LP)
    
    return x - tf.nn.depthwise_conv2d_native(x, LP_tf, padding = 'SAME', strides = [1,1,1,1]) 
    
def guided_filter(data, r = 15, eps = 0.3, batch_size = 4, width = 128, height = 128, channel = 15):
    batch_q = np.zeros((batch_size, height, width, channel))
    #        eps = self.eps
    for i in range(batch_size):
        for j in range(channel):
            I = 0.5* data[i, :, :,j] + 0.5# / 255.0
            p = 0.5*data[i, :, :,j] + 0.5 # / 255.0
            ones_array = np.ones([height, width])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0)
            mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            cov_Ip = mean_Ip - mean_I * mean_p; # this is the covariance of (I, p) in each local patch.
            mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps) # Eqn. (5) in the paper;
            b = mean_p - a * mean_I; # Eqn. (6) in the paper;
            mean_a = cv2.boxFilter(a , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_b = cv2.boxFilter(b , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            q = mean_a * I + mean_b # Eqn. (8) in the paper;
            batch_q[i, :, :,j] = (q -0.5) / 0.5#* 255.0 #range from 0 to 255
    return batch_q    
    
    
    
    
    
    
    
    
    
    
    
    
    
    