import numpy as np
import tensorflow as tf
from ops.blocks import resblock_down_in, resblock_up_in
from pdb import set_trace as bp


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
        s2, s4, s8, s16, s32 = int(image_size / 2), int(image_size / 4), int(image_size / 8), int(image_size / 16), \
                                 int(image_size/32)
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
            conv1 = tf.contrib.layers.instance_norm(conv1, center=True,
                                                 scale=True, trainable=True,
                                                  scope='e_bn1',
                                                  reuse=reuse)
            conv1 = tf.nn.leaky_relu(conv1, 0.2)

            self.enc_0 = conv1
            # image_size * image_size
            res1 = resblock_down_in(conv1, gf_dim, gf_dim, "res1", reuse, is_train)
            # res1 = tf.layers.conv2d(inputs=res1, filters = gf_dim*2, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res1_downsample')
            self.enc_res1=res1

            # s2*s2
            res2 = resblock_down_in(res1, gf_dim,gf_dim * 2, "res2", reuse, is_train)
            # res2 = tf.layers.conv2d(inputs=res2, filters = gf_dim*4, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res2_downsample')
            self.enc_res2=res2
            # s4*s4
            res3 = resblock_down_in(res2, gf_dim * 2, gf_dim * 4, "res3", reuse, is_train)
            # res3 = tf.layers.conv2d(inputs=res3, filters = gf_dim*8, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res3_downsample')
            self.enc_res3=res3
            # s8*s8
            res4 = resblock_down_in(res3, gf_dim * 4, gf_dim * 8, "res4", reuse, is_train)
            # res4 = tf.layers.conv2d(inputs=res4, filters = gf_dim*16, kernel_size = (ft_size,ft_size), strides=(2,2),
            #                    padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2), kernel_initializer = w_init,
            #                        trainable=True, name='res4_downsample')
            self.enc_res4=res4
            # s16*s16
            #res5 = resblock_down_in(res4, gf_dim * 8, gf_dim * 16, "res5", reuse, is_train)
            #self._activation_value_res5=res5
            # s32*s32
            res6 = resblock_down_in(res4, gf_dim * 8, gf_dim*16,
                                          "res6", reuse, is_train, act=False)
            #res6 = tf.reshape(res6, [-1, s32*s32, gf_dim*16])
            res6_std = resblock_down_in(res4, gf_dim * 8, gf_dim*16,
                                              "res6_std", reuse, is_train, act=False)

            res6_A = tf.layers.conv2d_transpose(res6_std, gf_dim*16, (3, 3), (2, 2), padding='same',
                                                kernel_initializer=w_init,
                                                bias_initializer=b_init, trainable=True, name="res6_A1",
                                                reuse=reuse)
            res6_A = tf.layers.conv2d_transpose(res6_A, gf_dim * 16, (3, 3), (2, 2), padding='same',
                                                kernel_initializer=w_init,
                                                bias_initializer=b_init, trainable=True, name="res6_A2",
                                                reuse=reuse)
            # res6_A = tf.layers.conv2d_transpose(res6_A, gf_dim * 16, (3, 3), (2, 2), padding='same',
            #                                     kernel_initializer=w_init,
            #                                     bias_initializer=b_init, trainable=True, name="res6_A3",
            #                                     reuse=reuse)

            res6 = tf.transpose(res6, [0, 3, 1, 2])
            res6_A = tf.transpose(res6_A, [0, 3, 1, 2])

            n = tf.shape(res6)[0]
            c = tf.shape(res6)[1]
            h = tf.shape(res6)[2]

            res6 = tf.reshape(res6, (n, c, 1, h*h))
            res6_A = tf.reshape(res6_A, (n, c, h * h, h * h))
            #res6_A = tf.exp(res6_A)
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

            conv2 = tf.contrib.layers.instance_norm(conv2,  center=True, scale=True,
                                                 trainable=True,
                                                 scope='e_bn2',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)

            conv2 = tf.layers.conv2d(conv2, gf_dim*2, (3,3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True,name="e_conv3",
                                     reuse=reuse)

            conv2 = tf.contrib.layers.instance_norm(conv2,  center=True, scale=True,
                                                 trainable=True,
                                                  scope='e_bn3',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)

            conv2 = tf.layers.conv2d(conv2, gf_dim, (3,3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="e_conv4",
                                     reuse=reuse)

            conv2 = tf.contrib.layers.instance_norm(conv2,  center=True, scale=True,
                                                 trainable=True,
                                                  scope='e_bn4',
                                                  reuse=reuse)
            conv2 = tf.nn.leaky_relu(conv2, 0.2)

            conv2 = tf.layers.conv2d(conv2, 1, (3,3), dilation_rate=2, padding='same',
                                     kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="e_conv5",
                                     reuse=reuse)
        return res6, res6_A, conv2, res4

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

            x = tf.reshape(x, [self.batchsize, gf_dim*16, s32, s32])
            x = tf.transpose(x, perm=[0,2,3,1])

            # 1*1
            #z_develop = tf.reshape(x, [-1, s64,s64, gf_dim*32])
            #z_develop = InputLayer(z_develop, name='g_inputs')
            #conv0 = DeConv2d(z_develop, gf_dim*32, (2, 2), (2, 2), (1, 1), act=lambda x: tl.act.lrelu(x, 0.2),
            #                padding="VALID", batch_size=batch_size, W_init=w_init, b_init=b_init, name="deconv0")

            #resp1 = resblock_up_in(x, gf_dim*32, gf_dim * 16, "gresp1", reuse, is_train)

            # s32*s32
            res0 = resblock_up_in(x, gf_dim*16, gf_dim * 8, "gres0", reuse, is_train)
            self.dec_res4 = res0
            #s16*s16
            # res0 = ResBlockUp(resp2, s16, batch_size, gf_dim * 16, "gres0", reuse, is_train)

            # s16*s16
            res1 = resblock_up_in(res0, gf_dim * 8, gf_dim * 4, "gres1", reuse, is_train)
            # res1 = tf.layers.conv2d_transpose(inputs=res1, filters = gf_dim*4, kernel_size = (ft_size, ft_size), strides = (2,2),
            #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
            #                                 kernel_initializer=w_init, trainable=True, name='res1_upsample')
            self.dec_res3 = res1
            # s8*s8
            res2 = resblock_up_in(res1,  gf_dim * 4, gf_dim * 2, "gres2", reuse, is_train)
            # res2 = tf.layers.conv2d_transpose(inputs=res2, filters=gf_dim*2, kernel_size=(ft_size, ft_size), strides=(2, 2),
            #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
            #                                  kernel_initializer=w_init, trainable=True, name='res2_upsample')
            self.dec_res2 = res2
            # s4*s4
            res3 = resblock_up_in(res2, gf_dim * 2, gf_dim , "gres3", reuse, is_train)
            #res3_std = resblock_up_bilinear(res2, s4, batch_size, gf_dim * 4, gf_dim * 2, "gres3_std", reuse, is_train)
            # res3 = tf.layers.conv2d_transpose(inputs=res3, filters=gf_dim, kernel_size=(ft_size, ft_size), strides=(2, 2),
            #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
            #                                  kernel_initializer=w_init, trainable=True, name='res3_upsample')
            self.dec_res1 = res3
            # s2*s2
            res4 = resblock_up_in(res3, gf_dim, gf_dim, "gres4", reuse, is_train)
            #res4_std = resblock_up_bilinear(res3_std, s2, batch_size, gf_dim * 2, gf_dim, "gres4_std", reuse, is_train)
            # res4 = tf.layers.conv2d_transpose(inputs=res4, filters=8, kernel_size=(ft_size, ft_size), strides=(2, 2),
            #                                  padding='SAME', activation=lambda x: tl.act.lrelu(x, 0.2),
            #                                  kernel_initializer=w_init, trainable=True, name='res4_upsample')
            self.dec_0 = res4
            # image_size*image_size
            conv2 = tf.layers.conv2d(res4, 1, (3, 3), padding='same', kernel_initializer=w_init,
                                     bias_initializer=b_init, trainable=True, name="g_conv2",
                                     reuse=reuse)
            #conv2_std = tf.layers.conv2d(res4_std, 1, (3, 3), padding='same', kernel_initializer=w_init2,
            #                              bias_initializer=b_init, trainable=True, name="g_conv2_std",
            #                              reuse=reuse)

        return conv2, res4, res0 #conv2_std, res4