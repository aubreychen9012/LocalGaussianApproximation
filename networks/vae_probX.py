import numpy as np
import tensorflow as tf
from ops.blocks import resblock_down, resblock_up
from pdb import set_trace as bp

class VariationalAutoencoder():
    def __init__(self,model_name=None, image_size=32):
        self.model_name = model_name
        self.image_size = image_size
        self.z_mean = None
        self.z_stddev = None

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
            conv1 = tf.layers.conv2d(x, gf_dim, (3,3), padding='same',
                                     kernel_initializer=w_init,
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
            #self._conv1 = conv1
            # image_size * image_size
            res1 = resblock_down(conv1, gf_dim, gf_dim, "res1", reuse, is_train)
            #self._activation_value_res1=res1

            # s2*s2
            res2 = resblock_down(res1, gf_dim, gf_dim * 2, "res2", reuse, is_train)
            #self._activation_value_res2=res2
            # s4*s4
            res3 = resblock_down(res2, gf_dim * 2, gf_dim * 4, "res3", reuse, is_train)

            #self._activation_value_res3=res3
            # s8*s8
            res4 = resblock_down(res3, gf_dim * 4, gf_dim * 8, "res4", reuse, is_train)
            #self._activation_value_res4=res4
            # s16*s16
            res5 = resblock_down(res4, gf_dim * 8, gf_dim * 16, "res5", reuse, is_train)
            #self._activation_value_res5=res5
            # s32*s32
            res6 = resblock_down(res5, gf_dim * 16, gf_dim *32, "res6", reuse, is_train, act=False)
            res6_stddev = resblock_down(res5, gf_dim * 16, gf_dim * 32,
                                        "res6_stddev", reuse, is_train, act=False)

            # s64*s64
            # res7 = resblock_valid_enc(res6, gf_dim * 16, gf_dim * 32, "res7", reuse, is_train)

            # res7 = resblock_down(res6, gf_dim * 32, gf_dim*16,
            #                               "res7", reuse, is_train)

            # enc_mean = tf.layers.conv2d(res6, gf_dim*32, (3, 3), (1, 1),
            #                             padding='same', kernel_initializer=w_init,
            #                      bias_initializer=b_init, trainable=True,
            #                             name="enc_mean", reuse=reuse)
            #
            # enc_stddev = tf.layers.conv2d(res6, gf_dim*32, (3, 3), (1, 1),
            #                             padding='same', kernel_initializer=w_init,
            #                      bias_initializer=b_init, trainable=True,
            #                             name="enc_stddev", reuse=reuse)
            # 40 x 40

            #res6_res = ResBlockDown(res5, gf_dim * 16, gf_dim * 4, "res6_res", reuse, is_train)
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

            #conv7_mean_flat = tf.contrib.layers.flatten(res_mean)
            #conv7_std_flat = tf.contrib.layers.flatten(res_stddev)
            #conv7_mean_res = tf.contrib.layers.flatten(res6_res)
            self.z_mean = res6
            self.z_stddev = res6_stddev

        return res6, res6_stddev, conv2

    def decoder(self, x, name, reuse=False, is_train=True):
        """
        Decoder part of the autoencoder.
        :param x: input to the decoder
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """

        gf_dim = 8
        # Dimension of gen filters in first conv layer. [64]
        with tf.variable_scope(self.model_name+"_decoder_"+name, reuse=reuse):
            # tl.layers.set_name_reuse(reuse)
            w_init = tf.truncated_normal_initializer(stddev=0.01)
            b_init = tf.constant_initializer(value=0.0)
            #gamma_init = tf.random_normal_initializer(0.5, 0.01)
            gamma_init = tf.ones_initializer()
            # 1*1
            # z_develop = tf.reshape(x, [-1, dim, dim, gf_dim*10])
            # s64

            resp1 = resblock_up(x, gf_dim * 32, gf_dim * 16, "gresp1", reuse, is_train)
            # s32*s32
            res0 = resblock_up(resp1, gf_dim*16, gf_dim * 8, "gres0", reuse, is_train)

            res1 = resblock_up(res0, gf_dim * 8, gf_dim * 4, "gres1", reuse, is_train)

            res2 = resblock_up(res1, gf_dim * 4, gf_dim * 2, "gres2", reuse, is_train)

            res3 = resblock_up(res2, gf_dim * 2, gf_dim, "gres3", reuse, is_train)

            res4 = resblock_up(res3, gf_dim, gf_dim, "gres4", reuse, is_train)

            conv1 = tf.layers.conv2d(res4, gf_dim, (3, 3), (1, 1),
                                        padding='same', kernel_initializer=w_init,
                                        bias_initializer=b_init, trainable=True,
                                        name="g_conv1", reuse=reuse)
            conv1 = tf.layers.batch_normalization(conv1, center=True,
                                                  scale=True, trainable=True,
                                                  gamma_initializer=gamma_init,
                                                  training=is_train,
                                                  name='g_bn1',
                                                  reuse=reuse)
            conv1 = tf.nn.leaky_relu(conv1, 0.2)

            # res5 = resblock_up(res4, gf_dim, gf_dim, "gres5", reuse, is_train)

            conv2 = tf.layers.conv2d(conv1, 1, (3, 3), padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="g_conv2",
                                     reuse=reuse)
            conv2_std = tf.layers.conv2d(conv1, 1, (3, 3), padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="g_conv2_std",
                                     reuse=reuse)
            conv2_std = tf.minimum(conv2_std, np.log(1e4))
        return conv2, conv2_std

    def predictor(self, z, reuse=False, is_train=True):
        """
        Decoder part of the autoencoder.
        :param x: input to the decoder
        :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
        :return: tensor which should ideally be the input given to the encoder.
        """

        gf_dim = 8
        # Dimension of gen filters in first conv layer. [64]
        with tf.variable_scope("predictor" , reuse=reuse):
            # tl.layers.set_name_reuse(reuse)
            w_init = tf.truncated_normal_initializer(stddev=0.01)
            b_init = tf.constant_initializer(value=0.0)
            #gamma_init = tf.random_normal_initializer(0.5, 0.01)
            gamma_init = tf.ones_initializer()
            # 1*1
            # z_develop = tf.reshape(x, [-1, dim, dim, gf_dim*10])
            # s64
            #x = tf.concat([zmean, zstddev], axis=3)
            x = z

            resp1 = resblock_up(x, gf_dim * 32, gf_dim * 16, "pred_resp1", reuse, is_train)
            # s32*s32
            res0 = resblock_up(resp1, gf_dim * 16, gf_dim * 8, "pred_res0", reuse, is_train)

            res1 = resblock_up(res0, gf_dim * 8, gf_dim * 4, "pred_res1", reuse, is_train)

            res2 = resblock_up(res1, gf_dim * 4, gf_dim * 2, "pred_res2", reuse, is_train)

            res3 = resblock_up(res2, gf_dim * 2, gf_dim, "pred_res3", reuse, is_train)

            res4 = resblock_up(res3, gf_dim, gf_dim, "pred_res4", reuse, is_train)

            conv1 = tf.layers.conv2d(res4, gf_dim, (3, 3), (1, 1),
                                     padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True,
                                     name="pred_conv1", reuse=reuse)
            conv1 = tf.layers.batch_normalization(conv1, center=True,
                                                  scale=True, trainable=True,
                                                  gamma_initializer=gamma_init,
                                                  training=is_train,
                                                  name='pred_bn1',
                                                  reuse=reuse)
            conv1 = tf.nn.leaky_relu(conv1, 0.2)

            # res5 = resblock_up(res4, gf_dim, gf_dim, "gres5", reuse, is_train)

            conv2 = tf.layers.conv2d(conv1, 1, (3, 3), padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="pred_conv2",
                                     reuse=reuse)
        return conv2
