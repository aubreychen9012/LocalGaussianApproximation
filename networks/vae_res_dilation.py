import numpy as np
import tensorflow as tf
#import tensorlayer as tl
#from tensorlayer.layers import *
from pdb import set_trace as bp
#from tensorflow.image import ResizeMethod

def lrelu(x):
    #alpha=0.2
  return tf.nn.relu(x) - 0.2 * tf.nn.relu(-x)


class VariationalAutoencoder():
    def __init__(self,model_name=None, batch_size=64, image_size=32, z_dim=10):
        #self.recons_images = tf.placeholder(tf.float32, [None, 40 * 40])
        #self.ref_images = tf.placeholder(tf.float32, [None, 40 * 40])
        self.model_name = model_name
        self.batchsize = batch_size
        self.image_size = image_size
        self.z_dim = z_dim

    def encoder(self, x, reuse=False, is_train=True):
        """
        Encode part of the autoencoder.
        :param x: input to the autoencoder
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :return: tensor which is the hidden latent variable of the autoencoder.
        """
        image_size = self.image_size
        gf_dim = 8  # Dimension of gen filters in first conv layer. [64]
        with tf.variable_scope(self.model_name+"_encoder", reuse=reuse):
            # x,y,z,_ = tf.shape(input_images)
            w_init = tf.truncated_normal_initializer(stddev=0.01)
            b_init = tf.constant_initializer(value=0.0)
            gamma_init = tf.random_normal_initializer(0.5, 0.01)
            # inputs = InputLayer(x, name='e_inputs')
            conv1 = tf.layers.conv2d(x, gf_dim, kernel_size = (4,4),
                                     strides = (2,2),
                                     padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init,
                                     trainable=True,
                                     name="e_conv1",
                                     reuse=reuse)
            conv1 = tf.layers.batch_normalization(conv1, center=True,
                                                 scale=True, trainable=True,
                                                gamma_initializer=gamma_init,
                                                  training=is_train,
                                                 name='e_bn1',
                                                  reuse=reuse)
            conv1 = tf.nn.leaky_relu(conv1, 0.2)
            self._conv1 = conv1
            # image_size * image_size

            res1 = tf.layers.conv2d(inputs=conv1, filters = gf_dim*2,
                                    kernel_size = (4,4), strides=(2,2),
                                    padding='same',
                                    activation=lrelu,
                                    kernel_initializer = w_init,
                                    trainable=True, name='res1_downsample')
            res1 = tf.layers.batch_normalization(res1, center=True, scale=True,
                                          gamma_initializer=gamma_init,
                                          trainable=True, training=is_train,
                                          name='rbu_res1', reuse=reuse)

            self._activation_value_res1=res1

            res2 = tf.layers.conv2d(inputs=res1, filters = gf_dim*4,
                                    kernel_size = (3,3), strides=(1,1),
                                    padding='VALID', activation=lrelu,
                                    kernel_initializer = w_init,
                                    trainable=True, name='res2_downsample')
            res2 = tf.layers.batch_normalization(res2, center=True, scale=True,
                                          gamma_initializer=gamma_init,
                                          trainable=True, training=is_train,
                                          name='rbu_res2', reuse=reuse)
            self._activation_value_res2=res2

            res3 = tf.layers.conv2d(inputs=res2, filters = gf_dim*8,
                                    kernel_size = (3,3), strides=(1,1),
                                    padding='VALID', activation=lrelu,
                                    kernel_initializer = w_init,
                                    trainable=True, name='res3_downsample')
            res3 = tf.layers.batch_normalization(res3, center=True, scale=True,
                                          gamma_initializer=gamma_init,
                                          trainable=True, training=is_train,
                                          name='rbu_res3', reuse=reuse)
            self._activation_value_res3=res3

            res4 = tf.layers.conv2d(inputs=res3, filters = gf_dim*16, kernel_size = (3,3),
                                    strides=(1,1),
                                    padding='VALID', activation=lrelu,
                                    kernel_initializer = w_init,
                                    trainable=True, name='res4_downsample')
            res4 = tf.layers.batch_normalization(res4, center=True, scale=True,
                                          gamma_initializer=gamma_init,
                                          trainable=True, training=is_train,
                                          name='rbu_res4', reuse=reuse)
            self._activation_value_res4=res4
            # s16*s16
            #res5 = resblock_down_bilinear(res4, gf_dim * 8, gf_dim * 16, "res5", reuse, is_train,sampling=False)
            res5 = tf.layers.conv2d(inputs=res4, filters=gf_dim * 32, kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='VALID', activation=lrelu,
                                    kernel_initializer=w_init,
                                    trainable=True, name='res5_downsample')
            res5 = tf.layers.batch_normalization(res5, center=True, scale=True,
                                          gamma_initializer=gamma_init,
                                          trainable=True, training=is_train,
                                          name='rbu_res5', reuse=reuse)

            self._activation_value_res5=res5
            # s32*s32
            #res6 = resblock_down_bilinear(res5, gf_dim * 16, gf_dim *32, "res6", reuse, is_train, sampling=False)
            res6 = tf.layers.conv2d(inputs=res5, filters=gf_dim * 64, kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='VALID', activation=lrelu,
                                    kernel_initializer=w_init,
                                    trainable=True, name='res6_downsample')
            # res6 = tf.layers.batch_normalization(res6, center=True, scale=True,
            #                               gamma_initializer=gamma_init,
            #                               trainable=True, training=is_train,
            #                               name='rbu_res6', reuse=reuse)

            # enc_mean enc_stddev has output of 38x38
            enc_mean = tf.layers.conv2d(inputs=res6, filters=20, kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='VALID',
                                    kernel_initializer=w_init,
                                    trainable=True, name='encoder_mean')

            enc_stddev = tf.layers.conv2d(inputs=res6, filters=20, kernel_size=(3, 3),
                                    strides=(1, 1),
                                    padding='VALID',
                                    kernel_initializer=w_init,
                                    trainable=True, name='encoder_stddev')

            ## residual prediction
            conv2 = tf.layers.conv2d(conv1, gf_dim, (3,3), dilation_rate =2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="e_conv2",
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2,  center=True, scale=True,
                                                 trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='e_bn2',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
            conv2 = tf.layers.conv2d(conv2, gf_dim*2, (3,3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True,name="e_conv3",
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2,  center=True, scale=True,
                                                 trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='e_bn3',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
            conv2 = tf.layers.conv2d(conv2, gf_dim*4, (3,3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="e_conv4",
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2,  center=True, scale=True,
                                                 trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='e_bn4',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
            conv2 = tf.layers.conv2d(conv2, gf_dim*8, (3,3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="e_conv5",
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                                  trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='e_bn5',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)

            conv2 = tf.layers.conv2d_transpose(inputs=conv2, filters=1,
                                       kernel_size=(3,3), strides=(2, 2),
                                       padding='same',
                                       kernel_initializer=w_init, trainable=True,
                                       name='resid_output')
            # conv2 = BatchNormLayer(conv2, act=lrelu, is_train=is_train,
            #                        gamma_init=gamma_init, name='e_bn2')

            # s64*s64
            #res7 = ResBlockDown(res6, gf_dim * 32, gf_dim * 32, "res7", reuse, is_train)

            #conv7 = InputLayer(res6, name="conv6_inputs")
            #conv7_mean = Conv2d(conv7, gf_dim*32, (2, 2), (1, 1), act=tf.identity,
            #                     padding='VALID', W_init=tf.truncated_normal_initializer(stddev=.1),
            #                     b_init=b_init, name="conv7_mean")
            # conv7_std = Conv2d(conv7, gf_dim * 32, (2, 2), (1, 1), act=tf.identity,
            #                    padding='VALID', W_init=tf.truncated_normal_initializer(stddev=.1),
            #                b_init=b_init,name="conv7_std")
            conv7_mean_flat = tf.contrib.layers.flatten(enc_mean)
            conv7_std_flat = tf.contrib.layers.flatten(enc_stddev)
            #conv7_mean_res = tf.contrib.layers.flatten(res6_res)
        return conv7_mean_flat, conv7_std_flat, conv2

    def decoder(self, x, name, reuse=False, is_train=True):
        """
        Decoder part of the autoencoder.
        :param x: input to the decoder
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """
        image_size = self.image_size
        s2, s4, s8, s16, s32, s64 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16), \
                               int(image_size/32), int(image_size/64)
        gf_dim = 8 # Dimension of gen filters in first conv layer. [64]
        with tf.variable_scope(self.model_name+"_decoder_"+name, reuse=reuse):
            #tl.layers.set_name_reuse(reuse)
            w_init = tf.truncated_normal_initializer(stddev=0.01)
            b_init = tf.constant_initializer(value=0.0)
            gamma_init = tf.random_normal_initializer(1., 0.02)

            # 1*1
            z_develop = tf.reshape(x, [-1, 38, 38, 20])

            res1 = tf.layers.conv2d_transpose(inputs=z_develop, filters = gf_dim*64,
                                              kernel_size = (3,3), strides = (1,1),
                                              padding='VALID',
                                              activation=lrelu,
                                             kernel_initializer=w_init, trainable=True,
                                              name='res1_upsample')
            res1 = tf.layers.batch_normalization(res1, center=True, scale=True,
                                          gamma_initializer=gamma_init,
                                          trainable=True, training=is_train,
                                          name='rbu_res1', reuse=reuse)

            res2 = tf.layers.conv2d_transpose(inputs=res1,
                                              filters = gf_dim*32,
                                              kernel_size = (3,3), strides = (1,1),
                                              padding='VALID',
                                              activation=lrelu,
                                             kernel_initializer=w_init, trainable=True,
                                              name='res2_upsample')
            res2 = tf.layers.batch_normalization(res2, center=True, scale=True,
                                          gamma_initializer=gamma_init,
                                          trainable=True, training=is_train,
                                          name='rbu_res2', reuse=reuse)

            res3 = tf.layers.conv2d_transpose(inputs=res2, filters=gf_dim*16,
                                              kernel_size=(3,3), strides=(1,1),
                                              padding='VALID',
                                              activation=lrelu,
                                              kernel_initializer=w_init, trainable=True,
                                              name='res3_upsample')
            res3 = tf.layers.batch_normalization(res3, center=True, scale=True,
                                          gamma_initializer=gamma_init,
                                          trainable=True, training=is_train,
                                          name='rbu_res3', reuse=reuse)

            res4 = tf.layers.conv2d_transpose(inputs=res3, filters=gf_dim*8,
                                              kernel_size=(3,3), strides=(1,1),
                                              padding='VALID', activation=lrelu,
                                              kernel_initializer=w_init, trainable=True,
                                              name='res4_upsample')
            res4 = tf.layers.batch_normalization(res4, center=True, scale=True,
                                          gamma_initializer=gamma_init,
                                          trainable=True, training=is_train,
                                          name='rbu_res4', reuse=reuse)

            res5 = tf.layers.conv2d_transpose(inputs=res4, filters=gf_dim*4,
                                              kernel_size=(3,3), strides=(1,1),
                                              padding='valid',
                                              activation=lrelu,
                                              kernel_initializer=w_init, trainable=True,
                                              name='res5_upsample')
            res5 = tf.layers.batch_normalization(res5, center=True, scale=True,
                                          gamma_initializer=gamma_init,
                                          trainable=True, training=is_train,
                                          name='rbu_res5', reuse=reuse)

            res6 = tf.layers.conv2d_transpose(inputs=res5, filters=gf_dim * 2,
                                              kernel_size=(3,3), strides=(1, 1),
                                              padding='valid',
                                              activation=lrelu,
                                              kernel_initializer=w_init, trainable=True,
                                              name='res6_upsample')
            res6 = tf.layers.batch_normalization(res6, center=True, scale=True,
                                          gamma_initializer=gamma_init,
                                          trainable=True, training=is_train,
                                          name='rbu_res6', reuse=reuse)

            res7 = tf.layers.conv2d_transpose(inputs=res6, filters=gf_dim,
                                              kernel_size=(4,4), strides=(2,2),
                                              padding='same',
                                              activation=lrelu,
                                              kernel_initializer=w_init, trainable=True,
                                              name='res7_upsample')
            res7 = tf.layers.batch_normalization(res7, center=True, scale=True,
                                                 gamma_initializer=gamma_init,
                                                 trainable=True, training=is_train,
                                                 name='rbu_res7', reuse=reuse)
            conv1 = tf.layers.conv2d_transpose(inputs=res7, filters=gf_dim,
                                              kernel_size=(4,4), strides=(2,2),
                                              padding='same',
                                              activation=lrelu,
                                              kernel_initializer=w_init, trainable=True,
                                              name='conv1_upsample')
            conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
                                                 gamma_initializer=gamma_init,
                                                 trainable=True, training=is_train,
                                                 name='rbu_conv1', reuse=reuse)

            # image_size*image_size
            conv2 = tf.layers.conv2d(conv1, 1, (3,3), padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="g_output",
                                     reuse=reuse)
        return conv2, res7, #conv2_std, res4
