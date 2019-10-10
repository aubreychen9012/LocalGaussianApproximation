import numpy as np
import tensorflow as tf
from tensorflow.python.ops import nn
from ops.blocks import resblock_down, resblock_up, resblock_up_bilinear,resblock_up_adabn
from pdb import set_trace as bp

class VariationalAutoencoder():
    def __init__(self,model_name=None, image_size=32):
        self.model_name = model_name
        self.image_size = image_size
        self.z_mean = None
        self.z_stddev = None

    def MLP(self, inputs, n_output, name):
        dense = tf.layers.dense(inputs, n_output*2, name=name+"_dense1")
        dense = tf.nn.leaky_relu(dense, 0.2)
        dense = tf.layers.dense(dense, n_output*4, name=name+"_dense2")
        dense = tf.nn.leaky_relu(dense, 0.2)
        dense = tf.layers.dense(dense, n_output, name=name+"_dense3")
        return dense

    def instance_norm(self, inputs, inputs_latent, name):
        inputs_rank = inputs.shape.ndims
        n_outputs = np.int(inputs.shape[-1])
        n_batch = np.int(inputs.shape[0])
        inputs_latent_flatten = tf.layers.flatten(inputs_latent)
        gamma = self.MLP(inputs_latent_flatten, n_outputs, name+"_gamma")
        beta = self.MLP(inputs_latent_flatten, n_outputs, name+"_beta")
        gamma = tf.reshape(gamma, [n_batch, 1, 1, n_outputs])
        beta = tf.reshape(beta, [n_batch, 1, 1, n_outputs])
        moments_axes = list(range(inputs_rank))
        mean, variance = nn.moments(inputs, moments_axes, keep_dims=True)
        outputs = nn.batch_normalization(
            inputs, mean, variance, beta, gamma, 1e-6, name=name)
        return outputs

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
            #res6_stddev = resblock_down(res5, gf_dim * 16, gf_dim * 32,
            #                            "res6_stddev", reuse, is_train, act=False)

            conv_latent = tf.layers.conv2d(res6, gf_dim*32, (1,1), padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="latent_mu",
                                     reuse=reuse)
            conv_latent_std = tf.layers.conv2d(res6, gf_dim * 32, (1, 1), padding='same',
                                           kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="latent_std",
                                           reuse=reuse)
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
                                     bias_initializer=b_init, trainable=True, name="stddev_conv2",
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2,  center=True, scale=True,
                                                 trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='stddev_bn2',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
            conv2 = tf.layers.conv2d(conv2, gf_dim*2, (3,3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True,name="stddev_conv3",
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2,  center=True, scale=True,
                                                 trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='stddev_bn3',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
            conv2 = tf.layers.conv2d(conv2, gf_dim, (3,3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="stddev_conv4",
                                     reuse=reuse)
            conv2 = tf.layers.batch_normalization(conv2,  center=True, scale=True,
                                                 trainable=True, training=is_train,
                                                  gamma_initializer=gamma_init, name='stddev_bn4',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)
            conv2 = tf.layers.conv2d(conv2, 1, (3,3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="stddev_conv5",
                                     reuse=reuse)
            #conv2 = tf.nn.tanh(conv2)
            #conv2 = tf.maximum(conv2, np.log(0.25))
            #conv7_mean_flat = tf.contrib.layers.flatten(res_mean)
            #conv7_std_flat = tf.contrib.layers.flatten(res_stddev)
            #conv7_mean_res = tf.contrib.layers.flatten(res6_res)

        return conv_latent, conv_latent_std, conv2

    def decoder(self, x, mu, name, reuse=False, is_train=True):
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
            conv_latent = tf.layers.conv2d(x, gf_dim * 32, (1, 1), padding='same',
                                           kernel_initializer=w_init,
                                           bias_initializer=b_init, trainable=True, name="latent_z",
                                           reuse=reuse)
            latent_flatten = tf.layers.flatten(x)
            batch_size = np.int(latent_flatten.shape[0])
            resp1_beta = self.MLP(latent_flatten, 1, "resp1_beta")
            resp1_gamma = self.MLP(latent_flatten, 1, "resp1_gamam")
            resp1 = resblock_up_adabn(conv_latent, gf_dim * 32, gf_dim * 16, "gresp1", reuse, is_train,
                                      gamma=resp1_gamma, beta=resp1_beta)
            #mean, variance = nn.moments(inputs, moments_axes, keep_dims=True)
            # s32*s32
            res0_beta = self.MLP(latent_flatten, 1, "res0_beta")
            res0_gamma = self.MLP(latent_flatten, 1, "res0_gamam")
            res0 = resblock_up_adabn(resp1, gf_dim*16, gf_dim * 8, "gres0", reuse, is_train,
                                     gamma=res0_gamma, beta=res0_beta)
            res1_beta = self.MLP(latent_flatten, batch_size, "res1_beta")
            res1_gamma = self.MLP(latent_flatten, batch_size, "res1_gamam")
            res1 = resblock_up_adabn(res0, gf_dim * 8, gf_dim * 4, "gres1", reuse, is_train,
                                     gamma=res1_gamma,beta=res1_beta)
            res2_beta = self.MLP(latent_flatten, 1, "res2_beta")
            res2_gamma = self.MLP(latent_flatten, 1, "res2_gamam")
            res2 = resblock_up_adabn(res1, gf_dim * 4, gf_dim * 2, "gres2", reuse, is_train,
                               gamma=res2_gamma, beta=res2_beta)
            res3_beta = self.MLP(latent_flatten, 1, "res3_beta")
            res3_gamma = self.MLP(latent_flatten, 1, "res3_gamam")
            res3 = resblock_up_adabn(res2, gf_dim * 2, gf_dim, "gres3", reuse, is_train,
                               gamma=res3_gamma, beta=res3_beta)
            res4_beta = self.MLP(latent_flatten, 1, "res4_beta")
            res4_gamma = self.MLP(latent_flatten, 1, "res4_gamam")
            res4 = resblock_up_adabn(res3, gf_dim, gf_dim, "gres4", reuse, is_train,
                               gamma=res4_gamma, beta=res4_beta)
            # res4 = tf.concat([res4, mask], axis=3)
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
            # conv1 = tf.concat([conv1, mask], axis=3)
            # res5 = resblock_up(res4, gf_dim, gf_dim, "gres5", reuse, is_train)

            # conv2 = tf.layers.conv2d(conv1, gf_dim/2, (3, 3), padding='same', kernel_initializer=w_init,
            #                          bias_initializer=b_init, trainable=True, name="g_conv2",
            #                          reuse=reuse)
            # conv2 = tf.nn.leaky_relu(conv2, 0.2)
            # conv2 = tf.layers.conv2d(conv2, gf_dim/4, (3, 3), padding='same', kernel_initializer=w_init,
            #                          bias_initializer=b_init, trainable=True, name="g_conv3_mu",
            #                          reuse=reuse)
            # conv2 = tf.nn.leaky_relu(conv2, 0.2)
            conv2 = tf.layers.conv2d(conv1, 1, (3, 3), padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="g_conv4_mu",
                                     reuse=reuse)

            # conv2_std = tf.layers.conv2d(conv1, 1, (3, 3), padding='same', kernel_initializer=w_init,
            #                          bias_initializer=b_init, trainable=True, name="g_conv2_std",
            #                          reuse=reuse)
            # conv2_std = tf.nn.leaky_relu(conv2_std, 0.2)
            # conv2_std = tf.layers.conv2d(conv2_std, 1, (1, 1), padding='same', kernel_initializer=w_init,
            #                              bias_initializer=b_init, trainable=True, name="g_conv3_std",
            #                              reuse=reuse)
            # conv2_std = tf.nn.leaky_relu(conv2_std, 0.2)
            # conv2_std = tf.layers.conv2d(conv2_std, 1, (1, 1), padding='same', kernel_initializer=w_init,
            #                              bias_initializer=b_init, trainable=True, name="g_conv4_std",
            #                              reuse=reuse)
            # conv2_std = tf.maximum(conv2_std, np.log(np.sqrt(1e-4)))
        return conv2 #, conv2_std