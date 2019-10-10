import os
import tensorflow as tf
import numpy as np
from utils import losses
from utils.batches import plot_batch
from pdb import set_trace as bp

class VAEModel():
    def __init__(self, model, config, model_name, log_dir):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        #self.image_mask = tf.placeholder('float32',
        #                                   [config["batch_size"], config["spatial_size"], config["spatial_size"], 1],
        #                                   name='mask')
        self.weight = tf.placeholder("float32", name='kl_weight')
        self.model = model()
        self.model.__init__(model_name, image_size = config["spatial_size"])
        self.model_name = model_name
        self.config = config
        self.loss = None
        self.train_op = None
        self.summary_op = None
        self.log_dir = log_dir
        self.writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
        self.build_network()
        self.saver = tf.train.Saver()


    def build_network(self):
        self.image_matrix = tf.placeholder('float32',
                                           [self.config["batch_size"], self.config["spatial_size"], self.config["spatial_size"], 1],
                                           name='input')
        self.z_mean, self.z_std, self.res = self.model.encoder(self.image_matrix, is_train=True, reuse=False)
        #z_stddev = tf.matmul(z_A, tf.transpose(z_A))
        self.n_channels = tf.shape(self.z_mean)[-1]
        samples = tf.random_normal(tf.shape(self.z_mean), 0., 1., dtype=tf.float32)
        self.guessed_z = self.z_mean + self.z_std*samples

        self.decoder_mean = self.model.decoder(self.guessed_z, name="img", is_train=True, reuse=False)
        #self.decoder_std = self.model.predictor(self.guessed_z, is_train=True, reuse=False)

        z_mean_valid, z_std_valid, self.res_test = self.model.encoder(self.image_matrix, is_train=False, reuse=True)
        samples_valid = tf.random_normal(tf.shape(z_mean_valid), 0., 1., dtype=tf.float32)
        guessed_z_valid = z_mean_valid + z_std_valid*samples_valid
        self.decoder_mean_test = self.model.decoder(guessed_z_valid, name="img", is_train=False, reuse=True)
        #self.decoder_std_test = self.model.predictor(guessed_z_valid, is_train=False, reuse=True)

        self.autoencoder_loss = losses.l2loss(self.image_matrix, self.decoder_mean)
        #self.negative_llh_loss = losses.negative_llh_var(self.image_matrix,
        #                                             self.decoder_mean, self.decoder_std)
        # self.autoencoder_loss = 5.*losses.l2loss(self.image_matrix,
        #                                                 self.decoder_mean)

        self.true_residuals = tf.abs(self.image_matrix-self.decoder_mean)
        self.autoencoder_res_loss = losses.l1loss(self.res, self.true_residuals)

        # 1d KL divergence
        self.latent_loss = losses.kl_loss_1d(self.z_mean, self.z_std)

        self.loss = tf.reduce_mean(20.*self.autoencoder_loss + self.weight*self.latent_loss + self.autoencoder_res_loss)

        ## validate
        self.autoencoder_loss_test = losses.l2loss(self.image_matrix, self.decoder_mean_test)
        #self.negative_llh_loss_test = losses.negative_llh_var(self.image_matrix, self.decoder_mean_test,
        #                                                     self.decoder_std_test)
        # self.autoencoder_loss_test = 5.*losses.l2loss(self.image_matrix, self.decoder_mean_test)
        self.true_residuals_test = tf.abs(self.image_matrix - self.decoder_mean_test)
        self.autoencoder_res_loss_test = losses.l1loss(self.res_test, self.true_residuals_test)
        self.latent_loss_test = losses.kl_loss_1d(z_mean_valid, z_std_valid)
        self.loss_test = tf.reduce_mean(20.*self.autoencoder_loss_test + self.weight*self.latent_loss_test
                                        + self.autoencoder_res_loss_test)

    # def initialize(self):
    #     t_vars = tf.trainable_variables()
    #     vae_vars = [var for var in t_vars if 'stddev_' not in var.name]
    #     stddev_vars = [var for var in t_vars if 'stddev_' in var.name]
    #     with tf.device("/gpu:0"):
    #         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #         with tf.control_dependencies(update_ops):
    #             # self.train_op = tf.train.AdamOptimizer(self.config["lr"]).minimize(self.loss)
    #             self.train_vae_op = tf.train.AdamOptimizer(self.config["lr"]).\
    #                 minimize(self.loss, var_list = vae_vars)
    #             self.train_stddev_op = tf.train.AdamOptimizer(self.config["lr"]).\
    #                 minimize(self.autoencoder_res_loss, var_list = stddev_vars)
    #     self.sess.run(tf.initializers.global_variables())

    def initialize(self):
        t_vars = tf.trainable_variables()
        #vae_vars = [var for var in t_vars if 'stddev_' not in var.name]
        #stddev_vars = [var for var in t_vars if 'stddev_' in var.name]
        with tf.device("/gpu:0"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # self.train_op = tf.train.AdamOptimizer(self.config["lr"]).minimize(self.loss)
                self.train_vae_op = tf.train.AdamOptimizer(self.config["lr"]).\
                    minimize(self.loss)
                #self.train_stddev_op = tf.train.AdamOptimizer(self.config["lr"]).\
                #    minimize(self.autoencoder_res_loss, var_list = stddev_vars)
        self.sess.run(tf.initializers.global_variables())

    def summarize(self):
        tf.summary.scalar("train_lat_loss", tf.reduce_mean(self.latent_loss))
        # tf.summary.scalar("test_gen_loss", tf.reduce_mean(autoencoder_loss_test))
        tf.summary.scalar("test_lat_loss", tf.reduce_mean(self.latent_loss_test))
        self.summary_op = tf.summary.merge_all()

    def train(self, input_images, weight):
        self.input_images = input_images
        feed_dict = {self.image_matrix: self.input_images,
                     self.weight: weight}
        self.sess.run(self.train_vae_op, feed_dict)

        self.out_mu = self.sess.run(self.decoder_mean, feed_dict)
        #self.out_std = self.sess.run(self.decoder_std, feed_dict)
        self.residual_output = self.sess.run(self.res, feed_dict)

    def train_var(self, input_images, weight):
        self.input_images = input_images
        feed_dict = {self.image_matrix: self.input_images,
                     self.weight: weight}
        self.sess.run(self.train_stddev_op, feed_dict)
        self.out_mu = self.sess.run(self.decoder_mean, feed_dict)
        #self.out_std = self.sess.run(self.decoder_std, feed_dict)
        self.residual_output = self.sess.run(self.res, feed_dict)

    def validate(self, input_images, weight):
        self.input_images_test = input_images
        feed_dict = {self.image_matrix: self.input_images_test,
                     self.weight:weight}

        self.out_mu_test = self.sess.run(self.decoder_mean_test, feed_dict)
        #self.out_std_test = self.sess.run(self.decoder_std_test, feed_dict)
        self.residual_output_test = self.sess.run(self.res_test, feed_dict)

    def visualize(self, model_name, ep):
        # out_mu[input_masks==0.]=-3.5
        # out_std = sess.run(decoder_std, {image_matrix:input_images})
        # out_std = np.exp(out_std)
        samples = self.sample()

        if not os.path.exists('Results/'+ model_name + '_samples/'):
            os.makedirs('Results/'+ model_name + '_samples/')
        model_name = 'Results/' + model_name
        plot_batch(self.input_images, model_name + '_samples/gr_' + str(ep) + '.png')
        plot_batch(self.out_mu, model_name + '_samples/gn_mu_' + str(ep) + '.png')
        #plot_batch(1./np.sqrt(np.exp(self.out_std)), model_name + '_samples/gn_std_' + str(ep) + '.png')
        #print((1./np.exp(self.out_std)).min(), (1./np.exp(self.out_std)).max())
        plot_batch(self.residual_output, model_name + '_samples/res_' + str(ep) + '.png')
        plot_batch(np.abs(self.input_images - self.out_mu), model_name + '_samples/gtres_' + str(ep) + '.png')
        plot_batch(samples, model_name + '_samples/generated_' + str(ep) + '.png')

        plot_batch(self.input_images_test, model_name + '_samples/test_gr_' + str(ep) + '.png')
        plot_batch(self.out_mu_test, model_name + '_samples/test_mu_' + str(ep) + '.png')
        #plot_batch(1./np.sqrt(np.exp(self.out_std_test)), model_name + '_samples/test_std_' + str(ep) + '.png')
        plot_batch(self.residual_output_test, model_name + '_samples/test_res_' + str(ep) + '.png')
        plot_batch(np.abs(self.input_images_test - self.out_mu_test), model_name + '_samples/test_gtres_' + str(ep) + '.png')

    def save(self,model_name, ep):
        if not os.path.exists(os.path.join(self.log_dir, model_name)):
            os.makedirs(os.path.join(self.log_dir, model_name))
        self.saver.save(self.sess, os.path.join(self.log_dir, model_name)+'/' + model_name + ".ckpt", global_step=ep)

    def load(self, model_name, step):
        model_folder = os.path.join(self.log_dir, model_name)
        self.saver.restore(self.sess, model_folder + '/' + model_name + ".ckpt-" + str(step))

    def sample(self):
        n_channels = self.sess.run(self.n_channels)
        mu = np.random.normal(0,1,(64,2,2, n_channels))
        #zeros = np.zeros((64,128,128,1))
        feed_dict = {self.guessed_z: mu}
        self.samples = self.sess.run(self.decoder_mean, feed_dict)
        return self.samples

