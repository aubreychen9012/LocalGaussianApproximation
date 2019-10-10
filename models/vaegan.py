import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SGE_GPU']
import tensorflow as tf
import yaml
import argparse
from pdb import set_trace as bp
from preprocess import *
import math
from batches import get_camcan_batches, tile, plot_batch
from vaegan import VAEGAN
from IndexFlow_camcan import IndexFlowCamCAN


def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)


class VaeganModel(VAEGAN):

    def __init__(self, model, config, model_name, log_dir):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.image_matrix = tf.placeholder('float32',
                                           [config["batch_size"], config["spatial_size"], config["spatial_size"], 1],
                                           name='input')
        self.model = model()
        self.model.__init__(model_name, image_size = config["spatial_size"])
        self.model_name = model_name
        self.lat_mu=0
        self.lat_stddev=0
        self.lat_samples=0
        self.sampled_z=0
        self.config = _config

        self.log_dir = log_dir
        self.writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)

        self.build_network()
        self.saver = tf.train.Saver()

    def network(self):
        self.batch_size = self.config["batch_size"]
        self.img_size = self.config["spatial_size"]
        self.image_matrix = tf.placeholder('float32',
                                           [self.batch_size,
                                            self.img_size, self.img_size, 1],
                                           name='input')
        self.lat_mu, self.lat_stddev = VAEGAN.encoder(self, self.image_matrix, is_train=True, reuse=False)
        self.lat_samples = tf.random_normal(tf.shape(self.lat_mu), 0, 1., dtype=tf.float32)
        self.sampled_z = self.lat_mu + (self.lat_stddev * self.lat_samples)
        self.decoder_output = VAEGAN.decoder(self,
            self.sampled_z, is_train=True, reuse=False)

        self.D, self.D_logits = VAEGAN.discriminator(self,self.image_matrix, reuse=False)
        self.D_, self.D_logits_ = VAEGAN.discriminator(self,self.decoder_output, reuse=True)

    def losses(self):
        w1 = 1.
        w2 = 1.
        w3 = 1.

        self.vae_recons_loss = tf.reduce_sum(tf.abs(self.decoder_output - self.image_matrix),
                                             [1, 2, 3])

        self.latent_loss = .5*tf.reduce_sum(
            tf.square(self.lat_mu) + tf.square(self.lat_stddev)
            - tf.log(tf.square(self.lat_stddev) + 1e-10) - 1, [1,2,3])

        self.d_loss_real = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real+self.d_loss_fake

        self.g_loss = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.vae_loss = tf.reduce_mean(self.latent_loss)+tf.reduce_mean(self.vae_recons_loss) #+tf.reduce_mean(self.latent_loss)

        self.gen_loss = w1*self.vae_recons_loss + w2*self.latent_loss + w3*self.g_loss
        self.dis_loss = self.d_loss

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'coder' in var.name]

    def build_network(self):
        self.network()
        self.losses()

    def initialize(self):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # self.vae_optim = tf.train.AdamOptimizer(self.config['lr']).minimize(self.vae_loss, var_list=self.g_vars)
            self.g_optim = tf.train.AdamOptimizer(self.config['lr']).minimize(self.gen_loss, var_list=self.g_vars)
            self.d_optim = tf.train.AdamOptimizer(self.config['lr']).minimize(self.dis_loss, var_list=self.d_vars)
            # self.optim = tf.train.AdamOptimizer(self.config['lr']).minimize(self.total_loss)

        self.sess.run(tf.initializers.global_variables())

    def train(self, input_images):
        #input_images, input_masks = next(batches)[:2]
        self.input_images = input_images.astype("float32")
        #input_masks = input_masks.astype("float32")
        feed_dict = {self.image_matrix: input_images}
        self.sess.run(self.g_optim, feed_dict)
        self.sess.run(self.d_optim, feed_dict)

    def validate(self, input_images):
        self.input_images_test = input_images
        feed_dict = {self.image_matrix: self.input_images_test}

        self.out_mu_test = self.sess.run(self.decoder_output_test, feed_dict)
        #self.residual_output_test = self.sess.run(self.res_test, feed_dict)

    def visualize(self, model_name, ep):
        if not os.path.exists('Results/' + model_name + '_samples/'):
            os.makedirs('Results/' + model_name + '_samples/')
        model_name = 'Results/' + model_name

        feed_dict = {self.image_matrix: self.input_images}
        self.out_mu = self.sess.run(self.decoder_output, feed_dict)

        plot_batch(self.input_images, model_name + '_samples/gr_' + str(ep) + '.png')
        plot_batch(self.out_mu, model_name + '_samples/gn_mu_' + str(ep) + '.png')
        # plot_batch(decoded_embedding[:, :, :, np.newaxis].astype("uint8"),
        #            os.path.join(wd, model_name + "_samples/embed_" + str(ep) + ".png"))
        plot_batch(self.residual_output, model_name + '_samples/res_' + str(ep) + '.png')
        plot_batch(np.abs(self.input_images - self.out_mu), model_name + '_samples/gtres_' + str(ep) + '.png')

        plot_batch(self.input_images_test, model_name + '_samples/test_gr_' + str(ep) + '.png')
        plot_batch(self.out_mu_test, model_name + '_samples/test_gn_mu_' + str(ep) + '.png')
        # plot_batch(decoded_embedding[:, :, :, np.newaxis].astype("uint8"),
        #            os.path.join(wd, model_name + "_samples/embed_" + str(ep) + ".png"))
        plot_batch(self.residual_output_test, model_name + '_samples/test_res_' + str(ep) + '.png')
        plot_batch(np.abs(self.input_images_test - self.out_mu_test),
                   model_name + '_samples/test_gtres_' + str(ep) + '.png')

    def save(self, model_name, ep):
        if not os.path.exists(os.path.join(self.log_dir, model_name)):
            os.makedirs(os.path.join(self.log_dir, model_name))
        self.saver.save(self.sess, os.path.join(self.log_dir, model_name) + '/' + model_name + ".ckpt",
                        global_step=ep)

    def load(self, model_name, step):
        model_folder = os.path.join(self.log_dir, model_name)
        self.saver.restore(self.sess, model_folder + '/' + model_name + ".ckpt-" + str(step))





        #
        # epochs = self.config['lr_decay_end']
        #
        # img_shape = 2 * [self.config["spatial_size"]] + [1]
        # data_shape = [self.batch_size] + img_shape
        # init_shape = [self.config["init_batches"] * self.batch_size] + img_shape
        # box_factor = self.config["box_factor"]
        # data_index = self.config["data_index"]
        # # z_dim = self.config["z_dim"]
        # # LR = self.config["lr"]
        # log_dir = self.config["log_dir"]
        # log_dir = os.path.join(log_dir, self._model_name)
        # out_dir = os.path.join("Results", self._model_name + '_samples')
        #
        # try:
        #     os.mkdir(log_dir)
        # except OSError:
        #     pass
        #
        # try:
        #     os.makedirs(out_dir)
        # except OSError:
        #     pass
        #
        # steps = 0
        # batches = get_camcan_batches(data_shape, data_index, train=True, box_factor=box_factor)
        # #init_batches = get_camcan_batches(init_shape, data_index, train=True, box_factor=box_factor)
        # valid_batches = get_camcan_batches(data_shape, data_index, train=False, box_factor=box_factor)
        #
        # for ep in range(epochs):
        #     input_images, input_masks = next(batches)[:2]
        #     input_images = input_images.astype("float32")
        #     feed_dict = {self.image_matrix: input_images}
        #     #self.sess.run(self.vae_optim, feed_dict)
        #     self.sess.run(self.g_optim, feed_dict)
        #     self.sess.run(self.d_optim, feed_dict)
        #     #self.sess.run(self.optim, feed_dict)
        #
        #     if ep % 500 == 0:
        #         recons_loss, lat_loss, d_loss \
        #             = self.sess.run(
        #             [self.vae_recons_loss, self.latent_loss,
        #               self.d_loss], feed_dict)
        #         print(("epoch %d: train_gen_loss %f train_lat_loss %f d_loss %f ") % (
        #             ep, recons_loss.mean(), lat_loss.mean(), d_loss.mean()))
        #
        #         out_mu = self.sess.run(self.decoder_output, feed_dict)
        #
        #         plot_batch(input_images, os.path.join(out_dir,'gr_' + str(ep) + ".png"))
        #         plot_batch(out_mu, os.path.join(out_dir,'gn_mu_' + str(ep) + ".png"))
        #
        #         saver.save(self.sess, os.path.join(log_dir, self._model_name + ".ckpt"), global_step=ep)
        #         images, masks = next(valid_batches)[:2]
        #         images = images.astype("float32")
        #         feed_dict = {self.image_matrix: images}
        #
        #         recons_loss_valid, lat_loss_valid, d_loss_valid,out \
        #             = self.sess.run([self.vae_recons_loss,
        #                                   self.latent_loss,
        #                                   self.d_loss,
        #                                   self.decoder_output],
        #                                  feed_dict)
        #         print("epoch %d: test_gen_loss %f test_lat_loss %f d_loss %f " % (
        #             ep, recons_loss_valid.mean(), lat_loss_valid.mean(), d_loss.mean()))
        #         plot_batch(images, os.path.join(out_dir,"test_gr" + str(ep) + ".png"))
        #         plot_batch(out, os.path.join(out_dir,"test_gn_mu_" + str(ep) + ".png"))