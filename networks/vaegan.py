import numpy as np
import tensorflow as tf
from utils.utils import resblock_down_bilinear, resblock_up_bilinear, lrelu, resblock_down, resblock_up
from pdb import set_trace as bp

#from tensorflow.image import ResizeMethod


class VAEGAN():
    def __init__(self, model_name=None):
        self.model_name = model_name

    def encoder(self, x, reuse=False, is_train=True):
        """
        Encode part of the autoencoder.
        :param x: input to the autoencoder
        :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
        :return: tensor which is the hidden latent variable of the autoencoder.
        """
        #image_size = self.image_size
        #h, w = x.shape[1:3]
        #h = int(h)
        #w = int(w)
        # s2, s4, s8, s16, s32 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16), \
        #                        int(image_size/32)
        gf_dim = 8  # Dimension of gen filters in first conv layer. [64]
        # c_dim = FLAGS.c_dim  # n_color 3
        # batch_size = 64  # 64
        with tf.variable_scope(self.model_name+"_encoder", reuse=reuse):
            # x,y,z,_ = tf.shape(input_images)
            w_init = tf.truncated_normal_initializer(stddev=0.01)
            b_init = tf.constant_initializer(value=0.0)
            gamma_init = tf.random_normal_initializer(0.5, 0.01)
            # inputs = InputLayer(x, name='e_inputs')
            conv1 = tf.layers.conv2d(x, gf_dim, (3,3), padding='same', kernel_initializer=w_init,
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
            conv1 = tf.nn.leaky_relu(conv1,0.2)
            self._conv1 = conv1
            # image_size * image_size
            res1 = resblock_down(conv1, gf_dim, gf_dim, "res1", reuse, is_train)
            # res1 = tf.layers.conv2d(inputs=res1, filters = gf_dim*2, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res1_downsample')
            self._activation_value_res1=res1

            # s2*s2
            res2 = resblock_down(res1, gf_dim,gf_dim * 2, "res2", reuse, is_train)
            # res2 = tf.layers.conv2d(inputs=res2, filters = gf_dim*4, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res2_downsample')
            self._activation_value_res2=res2
            # s4*s4
            res3 = resblock_down(res2, gf_dim * 2, gf_dim * 4, "res3", reuse, is_train)
            # res3 = tf.layers.conv2d(inputs=res3, filters = gf_dim*8, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res3_downsample')
            self._activation_value_res3=res3
            # s8*s8
            res4 = resblock_down(res3, gf_dim * 4, gf_dim * 8, "res4", reuse, is_train, act=False)
            res4_std = resblock_down(res3, gf_dim * 4, gf_dim * 8,
                                     "res4_std", reuse, is_train, act=False)
            # res4 = tf.layers.conv2d(inputs=res4, filters = gf_dim*16, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res4_downsample')
            #self._activation_value_res4=res4
            # s16*s16
            #res5 = resblock_down(res4, gf_dim * 8, gf_dim * 16, "res5", reuse, is_train, act=False)
            #res5_std = resblock_down(res4, gf_dim * 8, gf_dim * 16,
             #                        "res5_std", reuse, is_train, act=False)
            #self._activation_value_res5=res5
            # s32*s32
            #res6 = resblock_down(res5, gf_dim * 16, gf_dim*32,
            #                              "res6", reuse, is_train, act=False)
            #res6_std = resblock_down(res5, gf_dim * 16, gf_dim*32,
            #                                  "res6_std", reuse, is_train, act=False)
            #conv7_mean_flat = tf.contrib.layers.flatten(res6)
            #conv7_std_flat = tf.contrib.layers.flatten(res6_std)
        return res4, res4_std

    def decoder(self, x, reuse=False, is_train=True):
        """
        Decoder part of the autoencoder.
        :param x: input to the decoder
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """
        #image_size = self.image_size
        #s2, s4, s8, s16, s32, s64 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16), \
        #                       int(image_size/32), int(image_size/64)
        gf_dim = 8 # Dimension of gen filters in first conv layer. [64]
        with tf.variable_scope(self.model_name+"_decoder", reuse=reuse):
            #tl.layers.set_name_reuse(reuse)
            w_init = tf.truncated_normal_initializer(stddev=0.01)
            b_init = tf.constant_initializer(value=0.0)

            #resp1 = resblock_up(x, gf_dim*32, gf_dim * 16, "gresp1", reuse, is_train)

            # s32*s32
            #res0 = resblock_up(x, gf_dim*16, gf_dim * 16, "gres0", reuse, is_train)

            #s16*s16
            # res0 = ResBlockUp(resp2, s16, batch_size, gf_dim * 16, "gres0", reuse, is_train)

            # s16*s16
            res1 = resblock_up(x, gf_dim * 8, gf_dim * 4, "gres1", reuse, is_train)
            # res1 = tf.layers.conv2d_transpose(inputs=res1, filters = gf_dim*4, kernel_size = (ft_size, ft_size), strides = (2,2),
            #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
            #                                 kernel_initializer=w_init, trainable=True, name='res1_upsample')

            # s8*s8
            res2 = resblock_up(res1,  gf_dim * 4, gf_dim * 2, "gres2", reuse, is_train)
            # res2 = tf.layers.conv2d_transpose(inputs=res2, filters=gf_dim*2, kernel_size=(ft_size, ft_size), strides=(2, 2),
            #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
            #                                  kernel_initializer=w_init, trainable=True, name='res2_upsample')

            # s4*s4
            res3 = resblock_up(res2, gf_dim * 2, gf_dim, "gres3", reuse, is_train)
            #res3_std = resblock_up_bilinear(res2, s4, batch_size, gf_dim * 4, gf_dim * 2, "gres3_std", reuse, is_train)
            # res3 = tf.layers.conv2d_transpose(inputs=res3, filters=gf_dim, kernel_size=(ft_size, ft_size), strides=(2, 2),
            #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
            #                                  kernel_initializer=w_init, trainable=True, name='res3_upsample')

            # s2*s2
            res4 = resblock_up(res3, gf_dim, gf_dim, "gres4", reuse, is_train)
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

        return conv2 #, res4, #conv2_std, res4


    def discriminator(self, x, reuse=False, is_train=True):
        """
        Decoder part of the autoencoder.
        :param x: input to the decoder
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """
        #image_size = self.image_size
        batch_size = x.shape[0]
        # s2, s4, s8, s16, s32, s64 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16), \
        #                        int(image_size/32), int(image_size/64)
        gf_dim = 8  # Dimension of gen filters in first conv layer. [64]

        with tf.variable_scope(self.model_name+"discriminator", reuse=reuse):
            #tl.layers.set_name_reuse(reuse)
            w_init = tf.truncated_normal_initializer(stddev=0.01)
            b_init = tf.constant_initializer(value=0.0)
            gamma_init = tf.random_normal_initializer(0.5, 0.01)
            conv2 = tf.layers.conv2d(x, gf_dim, (3, 3), dilation_rate=2, padding='valid',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="conv2",
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                                  trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='bn2',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
            conv2 = tf.layers.conv2d(conv2, gf_dim * 2, (3, 3), dilation_rate=2, padding='valid',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="conv3",
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                                  trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='bn3',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
            conv2 = tf.layers.conv2d(conv2, gf_dim, (3, 3), dilation_rate=2, padding='valid',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="conv4",
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2, center=True, scale=True,
                                                  trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='bn4',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
            conv2 = tf.layers.conv2d(conv2, 1, (3, 3), dilation_rate=2, padding='valid',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="conv5",
                                     reuse=reuse)

            conv2_flatten = tf.reshape(conv2, [batch_size, -1])
            h4 = tf.layers.dense(conv2_flatten, 1,
                                 kernel_initializer=w_init,
                                 bias_initializer=b_init,
                                 trainable=True,
                                 name = 'dis_output', reuse=reuse)
            return tf.nn.sigmoid(h4), h4
