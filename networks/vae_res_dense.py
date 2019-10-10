import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
# import input_data
#import matplotlib.pyplot as plt
#import os
#from scipy.misc import imsave as ims
import sys
sys.path.append("/scratch_net/bmicdl01/chenx/PycharmProjects/vae_cnn")
from utils import *
from ops import *


def ResBlock(inputs, filters,pad, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        input_layer = InputLayer(inputs, name='e_inputs')
        conv1 = Conv2d(input_layer, filters,(3,3), act=None, padding=pad, W_init=w_init, b_init=b_init, name="e_conv1")
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='e_bn1')
        conv2 = Conv2d(conv1, filters, (3, 3), act=None, padding=pad, W_init=w_init, b_init=b_init, name="e_conv2")
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                           gamma_init=gamma_init, name='e_bn2')
        conv_out = conv2.outputs+inputs
        conv_out = tf.nn.relu(conv_out)
    return conv_out

def ResBlockDown(inputs, filters, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        input_layer = InputLayer(inputs, name='inputs')
        conv1 = Conv2d(input_layer, filters, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                       name="conv1")
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn1')
        conv2 = Conv2d(conv1, filters * 2, (3, 3), act=None, padding='SAME', W_init=w_init, b_init=b_init, name="conv2")
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn2')

        conv3 = Conv2d(input_layer, filters * 2, (3, 3), (2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                       name="conv3")
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn3')

        conv_out = conv2.outputs + conv3.outputs
    return conv_out


# image size *2
def ResBlockUp(inputs, input_size, batch_size, filters, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        input_layer = InputLayer(inputs, name='inputs')
        conv1 = DeConv2d(input_layer, filters, (3, 3), (input_size * 2, input_size * 2), (2, 2),
                         batch_size=batch_size, act=None, padding='SAME',
                         W_init=w_init, b_init=b_init, name="deconv1")
        conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn1')
        conv2 = DeConv2d(conv1, filters / 2, (3, 3), (input_size * 2, input_size * 2), (1, 1), act=None, padding='SAME',
                         batch_size=batch_size, W_init=w_init, b_init=b_init, name="deconv2")
        conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn2')

        conv3 = DeConv2d(input_layer, filters / 2, (3, 3), (input_size * 2, input_size * 2), (2, 2), act=None,
                         padding='SAME',
                         batch_size=batch_size, W_init=w_init, b_init=b_init, name="conv3")
        conv3 = BatchNormLayer(conv3, act=lambda x: tl.act.lrelu(x, 0.2), is_train=phase_train,
                               gamma_init=gamma_init, name='bn3')

        conv_out = conv2.outputs + conv3.outputs
    return conv_out

class VariationalAutoencoder():
    def __init__(self,model_name=None, batchsize=64, image_size=32, z_dim=10):
        #self.recons_images = tf.placeholder(tf.float32, [None, 40 * 40])
        #self.ref_images = tf.placeholder(tf.float32, [None, 40 * 40])
        self.model_name = model_name
        if 'refine' in model_name:
            self.ref=True
        else:
            self.ref=False
        self.batchsize = batchsize
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
        s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16)
        gf_dim = 8  # Dimension of gen filters in first conv layer. [64]
        ft_size = 3
        # c_dim = FLAGS.c_dim  # n_color 3
        # batch_size = 64  # 64
        with tf.variable_scope(self.model_name+"_encoder", reuse=reuse):
            # x,y,z,_ = tf.shape(input_images)
            tl.layers.set_name_reuse(reuse)

            w_init = tf.truncated_normal_initializer(stddev=0.02)
            b_init = tf.constant_initializer(value=0.0)
            gamma_init = tf.random_normal_initializer(1., 0.01)

            inputs = InputLayer(x, name='e_inputs')
            conv1 = Conv2d(inputs, gf_dim, (ft_size, ft_size), act=lambda x: tl.act.lrelu(x, 0.2), padding='SAME',
                           W_init=w_init, b_init=b_init,
                           name="e_conv1")
            conv1 = BatchNormLayer(conv1, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
                                   gamma_init=gamma_init, name='e_bn1')
            # image_size * image_size
            res1 = ResBlockDown(conv1.outputs, gf_dim, "res1", reuse, is_train)
            # res1 = tf.layers.conv2d(inputs=res1, filters = gf_dim*2, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res1_downsample')

            # s2*s2
            res2 = ResBlockDown(res1, gf_dim * 2, "res2", reuse, is_train)
            # res2 = tf.layers.conv2d(inputs=res2, filters = gf_dim*4, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res2_downsample')

            # s4*s4
            res3 = ResBlockDown(res2, gf_dim * 4, "res3", reuse, is_train)
            # res3 = tf.layers.conv2d(inputs=res3, filters = gf_dim*8, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res3_downsample')

            # s8*s8
            res4 = ResBlockDown(res3, gf_dim * 8, "res4", reuse, is_train)
            # res4 = tf.layers.conv2d(inputs=res4, filters = gf_dim*16, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res4_downsample')

            # s16*s16
            h_flat = tf.reshape(res4, shape=[-1, s16 * s16 * gf_dim * 16])
            h_flat = InputLayer(h_flat, name='e_reshape')
            net_h = DenseLayer(h_flat, n_units=self.z_dim, act=tf.identity, name="e_dense_mean")
            net_s = DenseLayer(h_flat, n_units=self.z_dim, act=tf.identity, name="e_dense_std")
        return net_h.outputs, net_s.outputs

    def decoder(self, x, reuse=False, is_train=True):
        """
        Decoder part of the autoencoder.
        :param x: input to the decoder
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """
        image_size = self.image_size
        s2, s4, s8, s16 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16)
        gf_dim = 8  # Dimension of gen filters in first conv layer. [64]
        c_dim = 1  # n_color 3
        ft_size = 3
        batch_size = self.batchsize # 64
        with tf.variable_scope(self.model_name+"_decoder", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            w_init = tf.truncated_normal_initializer(stddev=0.02)
            b_init = tf.constant_initializer(value=0.0)
            # gamma_init = tf.random_normal_initializer(1., 0.02)
            # weights_gener = dict()
            inputs = InputLayer(x, name='g_inputs')

            # s16*s16
            z_develop = DenseLayer(inputs, s16 * s16 * gf_dim * 16, act=lambda x: tl.act.lrelu(x, 0.2),
                                   name='g_dense_z')
            z_develop = tf.reshape(z_develop.outputs, [-1, s16, s16, gf_dim * 16])
            z_develop = InputLayer(z_develop, name='g_reshape')
            conv1 = Conv2d(z_develop, gf_dim * 8, (ft_size, ft_size), act=lambda x: tl.act.lrelu(x, 0.2),
                           padding='SAME',
                           W_init=w_init, b_init=b_init, name="g_conv1")

            # s16*s16
            res1 = ResBlockUp(conv1.outputs, s16, batch_size, gf_dim * 8, "gres1", reuse, is_train)
            # res1 = tf.layers.conv2d_transpose(inputs=res1, filters = gf_dim*4, kernel_size = (ft_size, ft_size), strides = (2,2),
            #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
            #                                 kernel_initializer=w_init, trainable=True, name='res1_upsample')

            # s8*s8
            res2 = ResBlockUp(res1, s8, batch_size, gf_dim * 4, "gres2", reuse, is_train)
            # res2 = tf.layers.conv2d_transpose(inputs=res2, filters=gf_dim*2, kernel_size=(ft_size, ft_size), strides=(2, 2),
            #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
            #                                  kernel_initializer=w_init, trainable=True, name='res2_upsample')

            # s4*s4
            res3 = ResBlockUp(res2, s4, batch_size, gf_dim * 2, "gres3", reuse, is_train)
            # res3 = tf.layers.conv2d_transpose(inputs=res3, filters=gf_dim, kernel_size=(ft_size, ft_size), strides=(2, 2),
            #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
            #                                  kernel_initializer=w_init, trainable=True, name='res3_upsample')

            # s2*s2
            res4 = ResBlockUp(res3, s2, batch_size, gf_dim, "gres4", reuse, is_train)
            # res4 = tf.layers.conv2d_transpose(inputs=res4, filters=8, kernel_size=(ft_size, ft_size), strides=(2, 2),
            #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
            #                                  kernel_initializer=w_init, trainable=True, name='res4_upsample')
            # image_size*image_size
            res_inputs = InputLayer(res4, name='res_inputs')
            conv2 = Conv2d(res_inputs, c_dim, (ft_size, ft_size), act=None, padding='SAME', W_init=w_init,
                           b_init=b_init,
                           name="g_conv2")
            conv2_std = Conv2d(res_inputs, c_dim, (ft_size, ft_size), act=None, padding='SAME', W_init=w_init,
                               b_init=b_init,
                               name="g_conv2_std")

            # deconv1 = DeConv2d(res_inputs, c_dim, (3, 3), out_size=(image_size, image_size), strides=(1, 1),
            #                   padding="SAME", act=None, batch_size=batch_size, W_init=w_init, b_init=b_init,
            #                   name="g_mu_output")
            # deconv1 = DeConv2d(res_inputs, c_dim, (3, 3), out_size=(image_size, image_size), strides=(1, 1),
            #                   padding="SAME", act=None, batch_size=batch_size, W_init=w_init, b_init=b_init,
            #                   name="g_std_output")
            logits = conv1.outputs
        return conv2.outputs