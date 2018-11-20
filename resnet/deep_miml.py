from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Reshape, Permute, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D


MIML_FIRST_LAYER_NAME = "miml/first_layer"
MIML_CUBE_LAYER_NAME = "miml/cube"
MIML_TABLE_LAYER_NAME = "miml/table"
MIML_OUTPUT_LAYER_NAME = "miml/output"

# with keras version
def create_miml_model(base_model, L, K, name="miml"):
    """
    Arguments:
        base_model (Sequential):
            A Neural Network in keras form (e.g. VGG, GoogLeNet)
        L (int):
            number of labels
        K (int):
            number of sub categories
    """
    model = Sequential(layers=base_model.layers, name=name)

    # input: feature_map.shape = (n_bags, C, H, W)
    _, C, H, W = model.layers[-1].output_shape
    print("Creating miml... input feature_map.shape={},{},{}".format(C, H, W))
    n_instances = H * W

    # shape -> (n_bags, (L * K), n_instances, 1)
    model.add(Convolution2D(L  * K, 1, 1, name=MIML_FIRST_LAYER_NAME))
    # shape -> (n_bags, L, K, n_instances)
    model.add(Reshape((L, K, n_instances), name=MIML_CUBE_LAYER_NAME))
    # shape -> (n_bags, L, 1, n_instances)
    model.add(MaxPooling2D((K, 1), strides=(1, 1)))
    # softmax
    model.add(Reshape((L, n_instances)))
    model.add(Permute((2, 1)))
    model.add(Activation("softmax"))
    model.add(Permute((2, 1)))
    model.add(Reshape((L, 1, n_instances), name=MIML_TABLE_LAYER_NAME))
    # shape -> (n_bags, L, 1, 1)
    model.add(MaxPooling2D((1, n_instances), strides=(1, 1)))
    # shape -> (n_bags, L)
    model.add(Reshape((L,), name=MIML_OUTPUT_LAYER_NAME))
    return model



 # tensorflow version
 # ==========================
 # "4 38 38 64"
        print "begin the  miml training "
        # ===begin the miml================
        b=x.shape[0]
        print "b:",b
        h=x.shape[1]
        w=x.shape[2]
        c=x.shape[3]
        n_instances=h*w
        L=FLAGS.CLASS
        K=FLAGS.sub_concept_k
        MIML_FIRST_LAYER_NAME = "miml_first_layer"
        MIML_CUBE_LAYER_NAME = "miml_cube"
        MIML_TABLE_LAYER_NAME = "miml_table"
        MIML_OUTPUT_LAYER_NAME = "miml_output"
        # conv38*38*64===>38*38*100(5*20)

        x = Conv2d(x,100,k_h=1, k_w=1, strides=[1, 1, 1, 1], name=MIML_FIRST_LAYER_NAME,
                           is_train=is_train, BN=BN)
        print type(x)

        print  "conv1",x.shape
        # 4*5*20*(n_instaances)
        # shape -> (n_bags, L, K, n_instances)
        x=tf.reshape(x,[4,5,20,-1],name=MIML_CUBE_LAYER_NAME)


        # maxpooling
        # shape -> (n_bags, L, 1, n_instances)
        x=MaxPooling(x,ksize = [1, 1, 20, 1], strides = [1, 1, 20, 1],
               padding = 'SAME', name = 'MaxPooling3')
        # softmax
        # reshape(4,5,1444)
        x=tf.reshape(x,[4,5,-1])
        # transpose(4,1444,5)
        x=tf.transpose(x,perm=[0,2,1])
        # # activation softmax(4,1444,5)
        x=tf.nn.softmax(x)
        # # transpose(4,5,1444)
        x=tf.transpose(x,perm=[0,2,1])
        # # reshape(4,5,1444,1)
        x=tf.reshape(x,[4,5,1444,1],name=MIML_TABLE_LAYER_NAME)
        # # reshape->(batch,L,1,1)
        #x = tf.maximum()=>(4,5,1,1)
        x = MaxPooling(x, ksize=[1, 1, 4,1], strides=[1, 1, 4,1],
                       padding='SAME', name='MaxPooling4')
        x = MaxPooling(x, ksize=[1, 1, 19, 1], strides=[1, 1, 19, 1],
                       padding='SAME', name='MaxPooling5')
        x = MaxPooling(x, ksize=[1, 1, 19, 1], strides=[1, 1, 19,1],
                       padding='SAME', name='MaxPooling6')
        print "xx",x.shape

        x = linear(tf.reshape(x, [FLAGS.BATCH_SIZE, -1]), FLAGS.output, name=MIML_OUTPUT_LAYER_NAME)
        softmax_x = tf.nn.softmax(x)
        print x,softmax_x