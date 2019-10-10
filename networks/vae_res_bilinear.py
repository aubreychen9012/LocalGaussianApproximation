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
def resblock_down(inputs, filters_in, filters_out, scope_name, reuse, phase_train):
    with tf.variable_scope(scope_name, reuse=reuse):
        #tl.layers.set_name_reuse(reuse)
        w_init = tf.truncated_normal_initializer(stddev=0.02)
        b_init = tf.constant_initializer(value=0.0)
        gamma_init = tf.random_normal_initializer(1., 0.02)
        #input_layer = InputLayer(inputs, name='inputs')
        conv1 = tf.layers.conv2d(inputs, filters_in, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True,name="rbd_conv1",reuse=reuse)
        conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True,
                                             gamma_initializer=gamma_init,
                                              trainable=True, training=phase_train, name='rbd_bn1',reuse=reuse)
        conv1 = tf.nn.leaky_relu(conv1, 0.2)
        conv2 = tf.layers.conv2d(conv1, filters_out, (3, 3), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True,name="rbd_conv2",reuse=reuse)
        conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbd_bn2',reuse=reuse)
        conv2 = tf.nn.leaky_relu(conv2, 0.2)
        conv3 = tf.layers.conv2d(inputs, filters_out, (3, 3), (2, 2), padding='same', kernel_initializer=w_init,
                                 bias_initializer=b_init, trainable=True, name="conv3",reuse=reuse)
        conv3 = tf.layers.batch_normalization(conv3, center=True, scale=True,
                                              trainable=True, training=phase_train,
                                              gamma_initializer=gamma_init, name='rbd_bn3', reuse=reuse)
        conv3 = tf.nn.leaky_relu(conv3, 0.2)
        conv_out = tf.add(conv2,conv3)
    return conv_out

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
        conv1 = tf.image.resize_images(conv1, tf.cast([h/2, w/2], tf.int32), method = tf.image.ResizeMethod.BILINEAR)
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
        input_identity = tf.image.resize_images(conv3, tf.cast([h / 2, w / 2], tf.int32), method=tf.image.ResizeMethod.BILINEAR)
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
        conv1 = tf.image.resize_images(conv1, tf.cast([h*2, w*2], tf.int32), method=tf.image.ResizeMethod.BILINEAR)
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
        input_identity = tf.image.resize_images(conv3, tf.cast([h * 2, w * 2], tf.int32), method=tf.image.ResizeMethod.BILINEAR)
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
        h, w = x.shape[1:3]
        h = int(h)
        w = int(w)
        # s2, s4, s8, s16, s32 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16), \
        #                         int(image_size/32)
        s64 = int(image_size/64)
        gf_dim = 8  # Dimension of gen filters in first conv layer. [64]
        # c_dim = FLAGS.c_dim  # n_color 3
        # batch_size = 64  # 64
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
            # res1 = tf.layers.conv2d(inputs=res1, filters = gf_dim*2, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res1_downsample')
            self._activation_value_res1=res1

            # s2*s2
            res2 = resblock_down_bilinear(res1, gf_dim,gf_dim * 2, "res2", reuse, is_train)
            # res2 = tf.layers.conv2d(inputs=res2, filters = gf_dim*4, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res2_downsample')
            self._activation_value_res2=res2
            # s4*s4
            res3 = resblock_down_bilinear(res2, gf_dim * 2, gf_dim * 4, "res3", reuse, is_train)
            # res3 = tf.layers.conv2d(inputs=res3, filters = gf_dim*8, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res3_downsample')
            self._activation_value_res3=res3
            # s8*s8
            res4 = resblock_down_bilinear(res3, gf_dim * 4, gf_dim * 8, "res4", reuse, is_train)
            # res4 = tf.layers.conv2d(inputs=res4, filters = gf_dim*16, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res4_downsample')
            self._activation_value_res4=res4
            # s16*s16
            res5 = resblock_down_bilinear(res4, gf_dim * 8, gf_dim * 16, "res5", reuse, is_train)
            self._activation_value_res5=res5
            # s32*s32
            res6 = resblock_down_bilinear(res5, gf_dim * 16, gf_dim*32,
                                          "res6", reuse, is_train, act=False)
            #res6 = tf.reshape(res6, [-1, s64*s64, gf_dim*32])
            res6_std = resblock_down_bilinear(res5, gf_dim * 16, gf_dim*32,
                                              "res6_std", reuse, is_train, act=False)

            res6_A = tf.layers.conv2d_transpose(res6_std, gf_dim*32, (3, 3), (2, 2), padding='same',
                                                kernel_initializer=w_init,
                                                bias_initializer=b_init, trainable=True, name="res6_A",
                                                reuse=reuse)

            res6 = tf.transpose(res6, [0, 3, 1, 2])
            res6_A = tf.transpose(res6_A, [0, 3, 1, 2])

            n = tf.shape(res6)[0]
            c = tf.shape(res6)[1]
            h = tf.shape(res6)[2]

            res6 = tf.reshape(res6, (n, c, 1, h*h))
            res6_A = tf.reshape(res6_A, (n, c, h * 2, h * 2))
            res6_A = tf.exp(res6_A)
            res6_A = tf.linalg.band_part(res6_A,-1, 0)


            # enc_mean = tf.layers.conv2d(res6, gf_dim*32, (3,3), padding='SAME',
            #                        kernel_initializer=w_init,
            #                        bias_initializer=b_init,
            #                        name ='enc_mean')
            # enc_stddev = tf.layers.conv2d(res6, gf_dim * 32, (3, 3), padding='SAME',
            #                            kernel_initializer=w_init,
            #                            bias_initializer=b_init,
            #                            name='enc_stddev')

            # concat1 = tf.image.resize_images(res1, tf.cast([h,w], tf.int32),
            #                                method=tf.image.ResizeMethod.BILINEAR)
            # concat2 = tf.image.resize_images(res2, tf.cast([h, w], tf.int32),
            #                                  method=tf.image.ResizeMethod.BILINEAR)
            # concat3 = tf.image.resize_images(res3, tf.cast([h, w], tf.int32),
            #                                  method=tf.image.ResizeMethod.BILINEAR)
            # concat4 = tf.image.resize_images(res4, tf.cast([h,w], tf.int32),
            #                                method=tf.image.ResizeMethod.BILINEAR)
            # concat5 = tf.image.resize_images(res5, tf.cast([h, w], tf.int32),
            #                                  method=tf.image.ResizeMethod.BILINEAR)
            # concat = tf.concat([x, concat1, concat2, concat3, concat4, concat5], axis=-1)

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

            # conv2 = BatchNormLayer(conv2, act=lambda x: tl.act.lrelu(x, 0.2), is_train=is_train,
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
            #conv7_mean_flat = tf.contrib.layers.flatten(res6)
            #conv7_A_flat = tf.contrib.layers.flatten(res6_A)


            #conv7_mean_res = tf.contrib.layers.flatten(res6_res)
        return res6, res6_A, conv2, res5

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
            #tl.layers.set_name_reuse(reuse)
            w_init = tf.truncated_normal_initializer(stddev=0.01)
            w_init2 = tf.truncated_normal_initializer(stddev=0.01)
            b_init = tf.constant_initializer(value=0.0)
            # gamma_init = tf.random_normal_initializer(1., 0.02)
            # weights_gener = dict()
            #inputs = InputLayer(x, name='g_inputs')

            x = tf.reshape(x, [self.batchsize, gf_dim*32, s64, s64])
            x = tf.transpose(x, perm=[0,2,3,1])

            # 1*1
            #z_develop = tf.reshape(x, [-1, s64,s64, gf_dim*32])
            #z_develop = InputLayer(z_develop, name='g_inputs')
            #conv0 = DeConv2d(z_develop, gf_dim*32, (2, 2), (2, 2), (1, 1), act=lambda x: tl.act.lrelu(x, 0.2),
            #                padding="VALID", batch_size=batch_size, W_init=w_init, b_init=b_init, name="deconv0")

            resp1 = resblock_up_bilinear(x, gf_dim*32, gf_dim * 16, "gresp1", reuse, is_train)

            # s32*s32
            res0 = resblock_up_bilinear(resp1, gf_dim*16, gf_dim * 16, "gres0", reuse, is_train)

            #s16*s16
            # res0 = ResBlockUp(resp2, s16, batch_size, gf_dim * 16, "gres0", reuse, is_train)

            # s16*s16
            res1 = resblock_up_bilinear(res0, gf_dim * 16, gf_dim * 8, "gres1", reuse, is_train)
            # res1 = tf.layers.conv2d_transpose(inputs=res1, filters = gf_dim*4, kernel_size = (ft_size, ft_size), strides = (2,2),
            #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
            #                                 kernel_initializer=w_init, trainable=True, name='res1_upsample')

            # s8*s8
            res2 = resblock_up_bilinear(res1,  gf_dim * 8, gf_dim * 4, "gres2", reuse, is_train)
            # res2 = tf.layers.conv2d_transpose(inputs=res2, filters=gf_dim*2, kernel_size=(ft_size, ft_size), strides=(2, 2),
            #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
            #                                  kernel_initializer=w_init, trainable=True, name='res2_upsample')

            # s4*s4
            res3 = resblock_up_bilinear(res2, gf_dim * 4, gf_dim * 2, "gres3", reuse, is_train)
            #res3_std = resblock_up_bilinear(res2, s4, batch_size, gf_dim * 4, gf_dim * 2, "gres3_std", reuse, is_train)
            # res3 = tf.layers.conv2d_transpose(inputs=res3, filters=gf_dim, kernel_size=(ft_size, ft_size), strides=(2, 2),
            #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
            #                                  kernel_initializer=w_init, trainable=True, name='res3_upsample')

            # s2*s2
            res4 = resblock_up_bilinear(res3, gf_dim * 2, gf_dim, "gres4", reuse, is_train)
            #res4_std = resblock_up_bilinear(res3_std, s2, batch_size, gf_dim * 2, gf_dim, "gres4_std", reuse, is_train)
            # res4 = tf.layers.conv2d_transpose(inputs=res4, filters=8, kernel_size=(ft_size, ft_size), strides=(2, 2),
            #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
            #                                  kernel_initializer=w_init, trainable=True, name='res4_upsample')
            # image_size*image_size
            conv2 = tf.layers.conv2d(res4, 1, (3, 3), padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="g_conv2",
                                     reuse=reuse)
            #conv2_std = tf.layers.conv2d(res4_std, 1, (3, 3), padding='same', kernel_initializer=w_init2,
            #                              bias_initializer=b_init, trainable=True, name="g_conv2_std",
            #                              reuse=reuse)

        return conv2, res4, resp1 #conv2_std, res4


    def att_encoder(self, x, reuse=False, is_train=True):
        """
        Encode part of the autoencoder.
        :param x: input to the autoencoder
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :return: tensor which is the hidden latent variable of the autoencoder.
        """
        image_size = self.image_size
        s2, s4, s8, s16, s32 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16), \
                               int(image_size/32)
        gf_dim = 8  # Dimension of gen filters in first conv layer. [64]
        ft_size = 3
        # c_dim = FLAGS.c_dim  # n_color 3
        # batch_size = 64  # 64
        with tf.variable_scope(self.model_name+"_encoder", reuse=reuse):
            # x,y,z,_ = tf.shape(input_images)
            #tl.layers.set_name_reuse(reuse)

            w_init = tf.truncated_normal_initializer(stddev=0.1)
            b_init = tf.constant_initializer(value=0.0)
            gamma_init = tf.random_normal_initializer(1., 0.01)

            #inputs = InputLayer(x, name='e_inputs')
            conv1 = tf.layers.conv2d(x, gf_dim, (3,3), padding='same', kernel_initializer=w_init, bias_initializer=b_init,
                                     trainable=is_train, name="e_conv1")
            conv1 = tf.layers.batch_normalization(conv1, center=True, scale=True, trainable=is_train,
                                   gamma_initializer=gamma_init, name='e_bn1')
            conv1 = tf.nn.leaky_relu(conv1, 0.2)
            # image_size * image_size
            #nlb1 = NonLocalBlock(conv1.outputs, gf_dim, is_bn=is_train, sub_sample=False, scope='nlb1')
            # res1 = tf.layers.conv2d(inputs=res1, filters = gf_dim*2, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res1_downsample')
            res1 = resblock_down(conv1, gf_dim, gf_dim * 2, "res1", reuse, is_train)
            # cnv1 = tf.layers.conv2d(nlb1,gf_dim*2, (3,3), strides=(1,1), name='Net_cnv1',
            #                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            #                         bias_initializer=tf.constant_initializer(value=0.0),padding='SAME')
            # cnv1_bn = tf.contrib.layers.batch_norm(cnv1, is_training=is_train, center=True, scale=True,
            #                                        scope='Net_cnv1_bn')
            # cnv1_pool = lrelu(cnv1_bn, alpha=0.2)

            # s2*s2
            #nlb2 = NonLocalBlock(res1, gf_dim * 2, is_bn=is_train, sub_sample=False,scope='nlb2')
            res2 = resblock_down(res1, gf_dim*2, gf_dim * 4, "res2", reuse, is_train)
            # cnv2 = tf.layers.conv2d(nlb2, gf_dim * 4,(3,3), strides=(1, 1), name='Net_cnv2',
            #                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            #                         bias_initializer=tf.constant_initializer(value=0.0),
            #                     padding='SAME')
            # cnv2_bn = tf.contrib.layers.batch_norm(cnv2, is_training=is_train,
            #                                        center=True, scale=True,
            #                                        scope='Net_cnv2_bn')
            # cnv2_pool = lrelu(cnv2_bn, alpha=0.2)
            #cnv2_pool = tf.nn.relu(tf.layers.max_pooling2d(cnv2_bn, (2, 2), strides=(1, 2, 2, 1), name='Net_cnv2_pool'))
            # res2 = tf.layers.conv2d(inputs=res2, filters = gf_dim*4, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res2_downsample')

            # s4*s4
            #nlb3 = NonLocalBlock(res2, gf_dim * 4, is_bn=is_train,sub_sample=False,scope="nlb3")
            res3 = resblock_down(res2, gf_dim*4, gf_dim * 8, "res3", reuse, is_train)
            # cnv3 = tf.layers.conv2d(nlb3, gf_dim * 8,(3,3), strides=(1, 1), name='Net_cnv3',
            #                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            #                         bias_initializer=tf.constant_initializer(value=0.0),
            #                     padding='SAME')
            # cnv3_bn = tf.contrib.layers.batch_norm(cnv3, is_training=is_train,
            #                                        center=True, scale=True,
            #                                         scope='Net_cnv3_bn')
            # cnv3_pool = lrelu(cnv3_bn, alpha=0.2)
            # res3 = tf.layers.conv2d(inputs=res3, filters = gf_dim*8, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res3_downsample')

            # s8*s8
            #nlb4 = NonLocalBlock(res3, gf_dim*8, is_bn=is_train, sub_sample=False,scope = "nlb4")
            #nlb4_std = NonLocalBlock(cnv3_pool, gf_dim*8, is_bn=is_train,scope="nlb4_std")
            res4 = resblock_down(res3, gf_dim*8, gf_dim*8, "res4", reuse, is_train)
           # nlb5 = NonLocalBlock(res4, gf_dim * 8, is_bn=is_train, sub_sample=False, scope="nlb5")
            #res5 = ResBlockDown(nlb5, gf_dim * 16, gf_dim * 16, "res5", reuse, is_train)

            # s32*s32
            res6 = resblock_down(res4, gf_dim * 8, gf_dim * 2, "res6", reuse, is_train)
            res6_std = resblock_down(res4, gf_dim * 8, gf_dim * 2, "res6_std", reuse, is_train)
            #res4_std = ResBlockDown(nlb4, gf_dim*8, gf_dim, "res1", reuse, is_train)

            # cnv4 = tf.layers.conv2d(nlb4, 2,(3,3), strides=(1, 1), name='Net_cnv4',
            #                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            #                         bias_initializer=tf.constant_initializer(value=0.0),
            #                     padding='SAME')
            # #cnv4_bn = tf.layers.batch_normalization(cnv4, gamma_initializer = tf.random_normal_initializer(1., 0.02),
            # #                                        name='Net_cnv4_bn')
            #
            # cnv4_std = tf.layers.conv2d(nlb4_std, 2, (3, 3), strides=(1, 1), name='Net_cnv4_std',
            #                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            #                        bias_initializer=tf.constant_initializer(value=0.0),
            #                        padding='SAME')
            #cnv4_bn_std = tf.layers.batch_normalization(cnv4_std, gamma_initializer=tf.random_normal_initializer(1., 0.02),
            #                                        name='Net_cnv4_bn_std')

            #cnv4_pool = lrelu(cnv4_bn, alpha=0.2)
            #cnv4_pool = tf.identity(cnv4_bn)

            conv7_mean_flat = tf.contrib.layers.flatten(res6)
            conv7_std_flat = tf.contrib.layers.flatten(res6_std)
        return conv7_mean_flat, conv7_std_flat


    def att_decoder(self, x, reuse=False, is_train=True):
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
        with tf.variable_scope(self.model_name+"_decoder", reuse=reuse):
            #tl.layers.set_name_reuse(reuse)
            w_init = tf.truncated_normal_initializer(stddev=0.1)
            b_init = tf.constant_initializer(value=0.0)

            # 1*1
            z_develop = tf.reshape(x, [-1, s32,s32, gf_dim*2])

            #resp1 = ResBlockUp(z_develop, s32, batch_size, gf_dim * 2, gf_dim * 8, "gresp1", reuse, is_train)

            # s16
            #nlb1 = NonLocalBlock(resp1, gf_dim * 16, sub_sample=False, is_bn=is_train,scope='g_nlb1')

            res1 = resblock_up(z_develop, s32, batch_size, gf_dim * 2, gf_dim * 8, "gres1", reuse, is_train)

            res2 = resblock_up(res1, s16, batch_size, gf_dim * 8, gf_dim * 4, "gres2", reuse, is_train)

            res3 = resblock_up(res2, s8, batch_size, gf_dim * 4, gf_dim*2, "gres3", reuse, is_train)

            res4 = resblock_up(res3, s4, batch_size, gf_dim * 2, gf_dim, "gres4", reuse, is_train)

            res5 = resblock_up(res4, s2, batch_size, gf_dim, gf_dim, "gres5", reuse, is_train)
            nlb5 = non_local_block(res5, gf_dim, sub_sample=False, is_bn=is_train, scope='g_nlb5')
            #res_inputs = InputLayer(nlb5, name='res_inputs')
            conv2 = tf.layers.conv2d(nlb5, c_dim, (1, 1), padding='same', kernel_initializer=w_init,
                           bias_initializer=b_init, trainable=is_train, name="g_conv2")
        return conv2, nlb5