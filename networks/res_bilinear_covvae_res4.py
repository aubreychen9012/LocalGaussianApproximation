import numpy as np
import tensorflow as tf
#import tensorlayer as tl
#from tensorlayer.layers import *
from pdb import set_trace as bp
#from tensorflow.image import ResizeMethod

def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def resblock(inputs, filters, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        #tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        #input_layer = InputLayer(inputs, name='e_inputs')
        conv1 = tf.layers.conv2d(inputs, filters, (3, 3), padding = 'same', kernel_initializer = w_init,
                                 bias_initializer=b_init, name="rb_conv1")
        conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
                                             gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train, name='rb_bn1')
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d(conv1, filters, (3, 3), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, name="rb_conv2")
        conv2 = tf.layers.batch_normalization(conv2,center=True, scale=True,
                                             gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train,name='rb_bn2')
        conv2 = tf.nn.leaky_relu(conv2, 0.2)
        conv_out = tf.add(conv2, inputs)
        #conv_out = tf.nn.relu(conv_out)
    return conv_out

def resblock_valid_enc(inputs, filters, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        #tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        #input_layer = InputLayer(inputs, name='e_inputs')
        conv1 = tf.layers.conv2d(inputs, filters, (3, 3), padding = 'valid', kernel_initializer = w_init,
                                 bias_initializer=b_init, name="rb_conv1")
        conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
                                             gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train, name='rb_bn1')
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d(conv1, filters, (3, 3), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, name="rb_conv2")
        conv2 = tf.layers.batch_normalization(conv2,center=True, scale=True,
                                             gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train,name='rb_bn2')
        conv2 = tf.nn.leaky_relu(conv2, 0.2)
        h,w = tf.shape(conv2)[1:3]
        inputs = tf.image.resize_images(inputs, tf.cast([h,w], tf.int32),
                                       method=tf.image.ResizeMethod.BILINEAR)
        conv_out = tf.add(conv2, inputs)
        #conv_out = tf.nn.relu(conv_out)
    return conv_out

# def resblock_valid_dec(inputs, filters, scope_name, reuse, phase_train):
#     with tf.variable_scope(scope_name, reuse=reuse):
#         #tl.layers.set_name_reuse(reuse)
#         w_init = tf.truncated_normal_initializer(stddev=0.02)
#         b_init = tf.constant_initializer(value=0.0)
#         gamma_init = tf.random_normal_initializer(1., 0.02)
#         #input_layer = InputLayer(inputs, name='e_inputs')
#         conv1 = tf.layers.conv2d_transpose(inputs, filters, (3, 3), padding = 'valid', kernel_initializer = w_init,
#                                  bias_initializer=b_init, name="rb_conv1")
#         conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
#                                              gamma_initializer=gamma_init,
#                                               trainable=True, training=phase_train, name='rb_bn1')
#         conv1 = tf.nn.leaky_relu(conv1, 0.2)
#         conv2 = tf.layers.conv2d(conv1, filters, (3, 3), padding='same', kernel_initializer=w_init,
#                                  bias_initializer=b_init, name="rb_conv2")
#         conv2 = tf.layers.batch_normalization(conv2,center=True, scale=True,
#                                              gamma_initializer=gamma_init,
#                                               trainable=True, training=phase_train,name='rb_bn2')
#         conv2 = tf.nn.leaky_relu(conv2, 0.2)
#         h,w = tf.shape(conv2)[1:3]
#         inputs = tf.image.resize_images(inputs, tf.cast([h,w], tf.int32),
#                                        method=tf.image.ResizeMethod.BILINEAR)
#         conv_out = tf.add(conv2, inputs)
#         #conv_out = tf.nn.relu(conv_out)
#     return conv_out
#
#
#
# def resblock_down(inputs, filters_in, filters_out, scope_name, reuse, phase_train):
#     with tf.variable_scope(scope_name, reuse=reuse):
#         #tl.layers.set_name_reuse(reuse)
#         w_init = tf.truncated_normal_initializer(stddev=0.02)
#         b_init = tf.constant_initializer(value=0.0)
#         gamma_init = tf.random_normal_initializer(1., 0.02)
#         #input_layer = InputLayer(inputs, name='inputs')
#         conv1 = tf.layers.conv2d(inputs, filters_in, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
#                                  bias_initializer=b_init, trainable=True,name="rbd_conv1",reuse=reuse)
#         conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
#                                              gamma_initializer=gamma_init,
#                                               trainable=True, training=phase_train, name='rbd_bn1',reuse=reuse)
#         conv1 = tf.nn.leaky_relu(conv1, 0.2)
#         conv2 = tf.layers.conv2d(conv1, filters_out, (3, 3), padding='same', kernel_initializer=w_init,
#                                  bias_initializer=b_init, trainable=True,name="rbd_conv2",reuse=reuse)
#         conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
#                                               trainable=True, training=phase_train,
#                                               gamma_initializer=gamma_init, name='rbd_bn2',reuse=reuse)
#         conv2 = tf.nn.leaky_relu(conv2, 0.2)
#         conv3 = tf.layers.conv2d(inputs, filters_out, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
#                                  bias_initializer=b_init, trainable=True, name="conv3",reuse=reuse)
#         conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
#                                               trainable=True, training=phase_train,
#                                               gamma_initializer=gamma_init, name='rbd_bn3', reuse=reuse)
#         conv3 = tf.nn.leaky_relu(conv3, 0.2)
#         conv_out = tf.add(conv2,conv3)
#     return conv_out

def resblock_down_bilinear(inputs, filters_in, filters_out,
                           scope_name, reuse, phase_train, act=True):
    h = tf.shape(inputs)[1]
    w = tf.shape(inputs)[2]
    with tf.variable_scope(scope_name, reuse=reuse):
        #tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.01)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        #input_layer = InputLayer(inputs, name='inputs')
        conv1 = tf.layers.conv2d(inputs, filters_in, (3, 3), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True,name="rbd_conv1",reuse=reuse)
        conv1 = tf.image.resize_images(conv1, tf.cast([h/2, w/2], tf.int32),
                                       method = tf.image.ResizeMethod.BILINEAR,
                                       align_corner=True)
        conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
                                             gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train, name='rbd_bn1',reuse=reuse)
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d(conv1, filters_out, (3, 3), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True,name="rbd_conv2",reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbd_bn2',reuse=reuse)
        conv2_leaky = tf.nn.leaky_relu(conv2, 0.2)
        conv3 = tf.layers.conv2d(inputs, filters_out, (3, 3), (1, 1), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True, name="conv3",reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbd_bn3', reuse=reuse)
        input_identity = tf.image.resize_images(conv3, tf.cast([h / 2, w / 2], tf.int32),
                                                method=tf.image.ResizeMethod.BILINEAR,
                                                align_corners=True)
        # conv3 = tf.nn.leaky_relu(conv3, 0.2)
        conv_out = tf.add(conv2,input_identity)
        if act:
            conv_out = tf.nn.leaky_relu(conv_out, 0.2)
    return conv_out

def resblock_up(inputs, input_size, batch_size, filters_in, filters_out, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        #tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        #input_layer = InputLayer(inputs, name='inputs')
        conv1 = tf.layers.conv2d_transpose(inputs, filters_in, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="rbu_deconv1",reuse=reuse)
        conv1 = tf.layers.batch_normalization(conv1, center=True,
                                             scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbu_bn1',reuse=reuse)
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d_transpose(conv1, filters_out, (3, 3), (1, 1), padding='same',
                        kernel_initializer=w_init, bias_initializer=b_init, trainable=True, name="rbu_deconv2",
                                           reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                             gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train,
                                              name='rbu_bn2',reuse=reuse)
        conv2 = tf.nn.leaky_relu(conv2, 0.2)
        conv3 = tf.layers.conv2d_transpose(inputs, filters_out, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                          bias_initializer=b_init, trainable=True, name="rbu_conv3",
                                           reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbu_bn3',
                                              reuse=reuse)
        conv3 = tf.nn.leaky_relu(conv3, 0.2)
        conv_out = tf.add(conv2, conv3)
    return conv_out


def resblock_up_bilinear(inputs, filters_in, filters_out,
                         scope_name, reuse, phase_train, act=True):
    h = tf.shape(inputs)[1]
    w = tf.shape(inputs)[2]
    with tf.variable_scope(scope_name, reuse=reuse):
        #tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.01)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        #input_layer = InputLayer(inputs, name='inputs')
        conv1 = tf.layers.conv2d_transpose(inputs, filters_in, (3, 3), (1, 1), padding='same', kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="rbu_deconv1",reuse=reuse)
        conv1 = tf.image.resize_images(conv1, tf.cast([h*2, w*2], tf.int32), method=tf.image.ResizeMethod.BILINEAR,
                                       align_corners=True)
        conv1 = tf.layers.batch_normalization(conv1, center=True,
                                             scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbu_bn1',reuse=reuse)
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d_transpose(conv1, filters_out, (3, 3), (1, 1), padding='same',
                        kernel_initializer=w_init, bias_initializer=b_init, trainable=True, name="rbu_deconv2",
                                           reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                             gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train,
                                              name='rbu_bn2',reuse=reuse)
        conv2_leaky = tf.nn.leaky_relu(conv2, 0.2)
        conv3 = tf.layers.conv2d_transpose(inputs, filters_out, (3, 3), (1, 1), padding='same', kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="rbu_conv3",
                                            reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbu_bn3',
                                              reuse=reuse)
        input_identity = tf.image.resize_images(conv3, tf.cast([h * 2, w * 2], tf.int32),
                                                method=tf.image.ResizeMethod.BILINEAR,
                                                align_corners =True)
        #conv3 = tf.nn.leaky_relu(conv3, 0.2)
        conv_out = tf.add(conv2, input_identity)
        if act:
            conv_out = tf.nn.leaky_relu(conv_out, 0.2)
    return conv_out


def non_local_block(input_x, out_channels, sub_sample=True, is_bn=False, scope='NonLocalBlock'):
    batchsize, height, width, in_channels = input_x.get_shape().as_list()
    with tf.variable_scope(scope) as sc:
        with tf.variable_scope('g') as scope:
            g = tf.layers.conv2d(input_x, out_channels, (1,1), strides=(1,1), padding ='SAME',
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 bias_initializer=tf.constant_initializer(value=0.0),
                                 name='g')
            if sub_sample:
                g= tf.image.resize_images(g, (height/2, width/2),
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                #g = tf.layers.max_pooling2d(g, (2,2), strides=(2,2), name='g_max_pool')

        with tf.variable_scope('phi') as scope:
            phi = tf.layers.conv2d(input_x, out_channels,(1,1), strides=(1,1), padding ='SAME',
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                   bias_initializer=tf.constant_initializer(value=0.0),
                                   name='phi')
            if sub_sample:
                phi=tf.image.resize_images(phi, (height/2, width/2),
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                #phi = tf.layers.max_pooling2d(phi, (2,2), strides=(2,2), name='phi_max_pool')

        with tf.variable_scope('theta') as scope:
            theta = tf.layers.conv2d(input_x, out_channels,(1,1), strides=(1,1), padding ='SAME',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                     bias_initializer=tf.constant_initializer(value=0.0),
                                     name='theta')
            if sub_sample:
                theta = tf.image.resize_images(theta, (height/2, width/2),
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                #theta = tf.layers.max_pooling2d(theta, (2,2), strides=(2,2), name='theta_max_pool')

        g_x = tf.reshape(g, [batchsize,out_channels, -1])
        g_x = tf.transpose(g_x, [0,2,1])

        theta_x = tf.reshape(theta, [batchsize, out_channels, -1])
        theta_x = tf.transpose(theta_x, [0,2,1])
        phi_x = tf.reshape(phi, [batchsize, out_channels, -1])

        f = tf.matmul(theta_x, phi_x)
        f_softmax = tf.nn.softmax(f, -1)
        y = tf.matmul(f_softmax, g_x)
        if sub_sample:
            height = height/2
            width=width/2
        y = tf.reshape(y, [batchsize, height, width, out_channels])
        with tf.variable_scope('w') as scope:
            w_y = tf.layers.conv2d(y, in_channels,(1,1), strides=(1,1), padding='SAME',
                                   kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                   bias_initializer=tf.constant_initializer(value=0.0),
                                   name ='y')
                #slim.conv2d(y, in_channels, [1,1], stride=1, scope='w')
            if is_bn:
                w_y = tf.layers.batch_normalization(w_y,center=True, scale=True,
                                                    trainable=True, training=is_bn,
                                                    name='nl_bn')
                    #slim.batch_norm(w_y)
        input_x = tf.image.resize_images(input_x, (height, width),
                                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        z = input_x + 0.5*w_y
        return z


class VariationalAutoencoder():
    def __init__(self,model_name=None, batch_size=64, image_size=32, z_dim=10):
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
        h, w = x.shape[1:3]
        h = int(h)
        w = int(w)
        s2, s4, s8, s16, s32 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16), \
                                 int(image_size/32)
        s64 = int(image_size/64)
        gf_dim = 8  # Dimension of gen filters in first conv layer. [64]

        with tf.variable_scope(self.model_name+"_encoder", reuse=reuse):
            # x,y,z,_ = tf.shape(input_images)
            w_init = tf.truncated_normal_initializer(stddev=0.02)
            b_init = tf.constant_initializer(value=0.0)
            gamma_init = tf.random_normal_initializer(1, 0.02)
            # inputs = InputLayer(x, name='e_inputs')
            conv1 = tf.layers.conv2d(x, gf_dim, (3,3), padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init,
                                     trainable=True,
                                     name="e_conv1",
                                     reuse=reuse)
            conv1 = tf.layers.batch_normalization(conv1, center=True,
                                                 scale=True, trainable=True,
                                                  training=is_train,
                                                 name='e_bn1',
                                                  reuse=reuse)
            conv1 = tf.nn.leaky_relu(conv1, 0.2)
            self._conv1 = conv1
            # image_size * image_size
            res1 = resblock_down_bilinear(conv1, gf_dim, gf_dim, "res1", reuse, is_train)

            # s2*s2
            res2 = resblock_down_bilinear(res1, gf_dim,gf_dim * 2, "res2", reuse, is_train)

            # s4*s4
            res3 = resblock_down_bilinear(res2, gf_dim * 2, gf_dim * 4, "res3", reuse, is_train)

            # s8*s8
            #res4 = resblock_down_bilinear(res3, gf_dim * 4, gf_dim * 8, "res4", reuse, is_train)

            # s16*s16
            # res5 = resblock_down_bilinear(res4, gf_dim * 8, gf_dim * 16, "res5", reuse, is_train)

            # s32*s32
            res6 = resblock_down_bilinear(res3, gf_dim * 4, gf_dim*8,
                                          "res6", reuse, is_train, act=False)
            #res6 = tf.reshape(res6, [-1, s64*s64, gf_dim*32])
            res6_std = resblock_down_bilinear(res3, gf_dim *4, gf_dim*8,
                                              "res6_std", reuse, is_train, act=False)

            # res6_A = tf.layers.conv2d_transpose(res6_std, gf_dim*8, (3, 3), (2, 2), padding='same',
            #                                     kernel_initializer=w_init,
            #                                     bias_initializer=b_init, trainable=True, name="res6_A",
            #                                     reuse=reuse)
            # res6_A = tf.layers.conv2d_transpose(res6_A, gf_dim * 8, (3, 3), (2, 2), padding='same',
            #                                     kernel_initializer=w_init,
            #                                     bias_initializer=b_init, trainable=True, name="res6_A_1",
            #                                     reuse=reuse)
            # res6_A = tf.layers.conv2d_transpose(res6_A, gf_dim * 8, (3, 3), (2, 2), padding='same',
            #                                     kernel_initializer=w_init,
            #                                     bias_initializer=b_init, trainable=True, name="res6_A_2",
            #                                     reuse=reuse)
            #
            # res6 = tf.transpose(res6, [0, 3, 1, 2])
            # res6_A = tf.transpose(res6_A, [0, 3, 1, 2])
            #
            # n = tf.shape(res6)[0]
            # c = tf.shape(res6)[1]
            # h = tf.shape(res6)[2]
            #
            # res6 = tf.reshape(res6, (n, c, 1, h*h))
            # res6_A = tf.reshape(res6_A, (n, c, h * h, h * h))
            # res6_A = tf.exp(res6_A)
            # res6_A = tf.linalg.band_part(res6_A,-1, 0)

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
            conv2 = tf.layers.conv2d(conv2, gf_dim, (3,3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="e_conv4",
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2,  center=True, scale=True,
                                                 trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='e_bn4',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
            conv2 = tf.layers.conv2d(conv2, 1, (3,3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="e_conv5",
                                     reuse=reuse)
        return res6, res6_std, conv2

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
        c_dim = 1  # n_color 3
        ft_size = 3
        batch_size = self.batchsize # 64
        with tf.variable_scope(self.model_name+"_decoder_"+name, reuse=reuse):
            w_init = tf.truncated_normal_initializer(stddev=0.01)
            b_init = tf.constant_initializer(value=0.0)

            #x = tf.reshape(x, [self.batchsize, gf_dim*8, s16, s16])
            #x = tf.transpose(x, perm=[0,2,3,1])

            #resp1 = resblock_up_bilinear(x, gf_dim*32, gf_dim * 16, "gresp1", reuse, is_train)

            # s32*s32
            #res0 = resblock_up_bilinear(resp1, gf_dim*16, gf_dim * 16, "gres0", reuse, is_train)

            # s16*s16
            res1 = resblock_up_bilinear(x, gf_dim * 8, gf_dim * 4, "gres1", reuse, is_train)

            # s8*s8
            res2 = resblock_up_bilinear(res1,  gf_dim * 4, gf_dim * 2, "gres2", reuse, is_train)

            # s4*s4
            res3 = resblock_up_bilinear(res2, gf_dim *2, gf_dim, "gres3", reuse, is_train)

            # s2*s2
            res4 = resblock_up_bilinear(res3, gf_dim, gf_dim, "gres4", reuse, is_train)

            # image_size*image_size
            conv2 = tf.layers.conv2d(res4, 1, (3, 3), padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="g_conv2",
                                     reuse=reuse)
        return conv2, res4

