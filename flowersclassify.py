# coding=utf8
from skimage import io, transform
import glob
import os
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from sklearn.utils import shuffle



# 数据集地址
path = 'flowers/'
path0 = "flowers/daisy/"
path1 = "flowers/dandelion/"
path2 = "flowers/rose/"
path3 = "flowers/sunflower/"
path4 = "flowers/tulip/"
path=[path0,path1,path2,path3,path4]

# 模型保存地址
model_path = 'model/model.ckpt'

# 将所有的图片resize成100*100
w = 100
h = 100
c = 1
moving_average_decay=0.99
import random

# 读取图片 one foler //resize  gray
def read_img(folder,idx):
    imgs=[]
    labels=[]
    for im in glob.glob(folder+'/*.jpg'):
        print('reading the images:%s'%(im))
        # gray
        img=io.imread(im,as_grey=True)
     #  img=io.imread(im)
        img=transform.resize(img,(w,h))
     # # shift
     #    shift = ndimage.shift(img, (5, 5))
     # #  rotare
     #    rotate = transform.rotate(img, 45, resize=False)
        imgs.append(img)
        labels.append(idx)
     #
     #    imgs.append(shift)
     #    labels.append(idx)
     #
     #    imgs.append(rotate)
     #    labels.append(idx)

    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)

# # 将所有数据分为训练集和验证集
ratio = 0.8
for i in range(2):

    data, label = read_img(path[i], i)
    num_example = data.shape[0]
    # shuffle the data
    data,label= shuffle(data, label, random_state=0)

    s = np.int(num_example * ratio)
    if i==0:
        x_train = data[:s]
        y_train = (label[:s])
        x_val = (data[s:])
        y_val = (label[s:])

    else:
        x_train=np.append(x_train, data[:s], axis=0)
        y_train=np.append(y_train, label[:s], axis=0)
        x_val=np.append(x_val, data[s:], axis=0)
        y_val=np.append(y_val, label[s:], axis=0)
[a,b,c_] = x_train.shape
[a_v,b_v,c_v]= x_val.shape

#
x_train = x_train.reshape(a,b,c_,1)
x_val = x_val.reshape(a_v,b_v,c_v,1)






#adad
print x_train.shape
print y_train.shape
print x_val.shape
print y_val.shape





# # 可视化前30张 6行5列
# plt.figure(figsize=(32,32))
# for i in range(0,30):
#     plt.subplot(6, 5, (i+1))
#     plt.imshow(x_train[i])
#     plt.title('label:'+str(y_train[i]))
# plt.show()












# -----------------构建网络----------------------
# 占位符
x = tf.placeholder(tf.float32, shape=[None, w, h,c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

#  no average
def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight", [5, 5, 1, 32],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight", [5, 5, 32, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight", [3, 3, 64, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable("weight", [3, 3, 128, 128],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        nodes = 6 * 6 * 128
        reshaped = tf.reshape(pool4, [-1, nodes])

    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [512, 5],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [5], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit


conv1_weights = tf.get_variable("weight1", [5, 5, 1, 32],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))

conv1_biases = tf.get_variable("bias1", [32], initializer=tf.constant_initializer(0.0))

conv2_weights = tf.get_variable("weight2", [5, 5, 32, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
conv2_biases = tf.get_variable("bias2", [64], initializer=tf.constant_initializer(0.0))


conv3_weights = tf.get_variable("weight3", [3, 3, 64, 128],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
conv3_biases = tf.get_variable("bias3", [128], initializer=tf.constant_initializer(0.0))

conv4_weights = tf.get_variable("weight4", [3, 3, 128, 128],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
conv4_biases = tf.get_variable("bias4", [128], initializer=tf.constant_initializer(0.0))



def inference_average(input_tensor, train, regularizer,avg_class):
    with tf.variable_scope('layer1-conv1'):
        # conv1_weights = tf.get_variable("weight", [5, 5, 1, 32],
        #                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        # conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        print conv1_weights
        print avg_class.average(conv1_weights)

        conv1 = tf.nn.conv2d(input_tensor, avg_class.average(conv1_weights), strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, avg_class.average(conv1_biases)))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    with tf.variable_scope("layer3-conv2"):
        # conv2_weights = tf.get_variable("weight", [5, 5, 32, 64],
        #                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        # conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, avg_class.average(conv2_weights), strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, avg_class.average(conv2_biases)))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer5-conv3"):
        # conv3_weights = tf.get_variable("weight", [3, 3, 64, 128],
        #                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        # conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, avg_class.average(conv3_weights), strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, avg_class.average(conv3_biases)))

    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer7-conv4"):
        # conv4_weights = tf.get_variable("weight", [3, 3, 128, 128],
        #                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        # conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, avg_class.average(conv4_weights), strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, avg_class.average(conv4_biases)))

    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        nodes = 6 * 6 * 128
        reshaped = tf.reshape(pool4, [-1, nodes])

    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [512, 5],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [5], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights + fc3_biases)

    return logit


# ---------------------------网络结束---------------------------
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
# logits = inference(x, False, regularizer)
# --------average
step = tf.Variable(0, trainable=False)
variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay,num_updates=step)
variable_averages_op = variable_averages.apply(tf.trainable_variables())
logits = inference_average(x, False, regularizer,variable_averages)

# -------------------------minibatch-----------------------------

# (小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
b = tf.constant(value=1, dtype=tf.float32)
logits_eval = tf.multiply(logits, b, name='logits_eval')

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# ---all add averages
# train_all=tf.group(train_op,variable_averages_op)
with tf.control_dependencies([train_op,variable_averages_op]):
    train_all=tf.no_op(name='train')
# ---all
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# 训练和测试数据，可将n_epoch设置更大一些

n_epoch = 30
batch_size = 64
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# graph --
writer=tf.summary.FileWriter("./graph/")
writer.add_graph(sess.graph)

for epoch in range(n_epoch):
    print "======epoch======",epoch
    start_time = time.time()

    # training
    train_loss, train_acc, n_batch = 0, 0, 0

    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):


        _, err, ac = sess.run([train_all, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err;
        train_acc += ac;
        n_batch += 1

    print "   train loss: %f" % (np.sum(train_loss) / n_batch)
    print "   train acc: %f" % (np.sum(train_acc) / n_batch)

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err;
        val_acc += ac;
        n_batch += 1
    print "   validation loss: %f" % (np.sum(val_loss) / n_batch)
    print "   validation acc: %f" % (np.sum(val_acc) / n_batch)



saver.save(sess, model_path)
sess.close()

