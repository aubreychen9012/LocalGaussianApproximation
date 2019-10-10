import os
import tensorflow as tf
import numpy as np
from utils import losses
from utils import deeploss
from utils.batches import plot_batch
import matplotlib.pyplot as plt
from pdb import set_trace as bp

class AdversarialAE():
    def __init__(self, model, config, model_name, log_dir):
        gpu_config = tf.ConfigProto(allow_soft_placement=True)
        gpu_config.gpu_options.allow_growth = True
        #gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(config=gpu_config)
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
                                           [self.config["batch_size"], self.config["spatial_size"],
                                            self.config["spatial_size"], 1],
                                           name='input')
        # self.z_mean, self.z_A_T, self.res = \
        #     self.model.encoder(self.image_matrix, is_train=True, reuse=False)

        self.z, self.res = self.model.encoder(self.image_matrix, is_train=True, reuse=False)
        self.z_real = tf.random_normal(tf.shape(self.z), mean=0.0, stddev=1.0)

        self.d_fake = self.model.discriminator(self.z, is_train=True, reuse=False)

        self.d_real = self.model.discriminator(self.z_real, is_train=True, reuse=True)

        #guessed_z = z_mean + (z_A_T * samples)
        self.decoder_output = \
            self.model.decoder(self.z, is_train=True, reuse=False)

        self.z_valid, self.res_test = \
            self.model.encoder(self.image_matrix, is_train=False, reuse=True)
        self.z_real_valid = tf.random_normal(tf.shape(self.z_valid), mean=0.0, stddev=1.0)

        self.decoder_output_test = \
            self.model.decoder(self.z_valid, is_train=False, reuse=True)

        self.d_real_test = self.model.discriminator(self.z_real_valid, is_train=True, reuse=True)

        self.d_fake_test = self.model.discriminator(self.z_valid, is_train=True, reuse=True)

        # losses
        self.autoencoder_loss = 5.*losses.l2loss(self.decoder_output, self.image_matrix)

        self.true_residuals = tf.abs(self.image_matrix - self.decoder_output)
        self.autoencoder_res_loss = losses.l2loss(self.res, self.true_residuals)

        self.l2_loss = tf.reduce_mean(self.autoencoder_loss+self.autoencoder_res_loss)

        tf_randn_real = tf.random_uniform(tf.shape(self.d_real), 0.9, 1.0)
        tf_randn_fake = tf.random_uniform(tf.shape(self.d_fake), 0.0, 0.1)

        dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_randn_real, logits=self.d_real))
        dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_randn_fake, logits=self.d_fake))
        self.dc_loss = dc_loss_fake + dc_loss_real

        # with tf.name_scope("Gradient_penalty"):
        #     eta = tf.random_uniform(self.config['batch_size'])
        #     interp = tf.multiply(eta,self.z_real) + tf.multiply((1-eta), self.z)
        #     _,c_interp = self.model.discriminator(interp, reuse=True)
        #
        #     # taking the zeroth and only element because tf.gradients returns a list
        #     c_grads = tf.gradients(c_interp, interp)[0]
        #
        #     # L2 norm, reshaping to [batch_size]
        #     slopes = tf.sqrt(tf.reduce_sum(tf.square(c_grads), axis=[1]))
        #     tf.summary.histogram("Critic gradient L2 norm", slopes)
        #
        #     grad_penalty = tf.reduce_mean((slopes - 1) ** 2)
        #     lambd = 10.0
        #     self.dc_loss += lambd * grad_penalty

        self.gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_fake),
                                                    logits=self.d_fake))

        ## validate
        self.autoencoder_loss_test = 5.*losses.l2loss(self.decoder_output_test, self.image_matrix)

        self.true_residuals_test = tf.abs(self.image_matrix - self.decoder_output_test)
        self.autoencoder_res_loss_test = losses.l2loss(self.res_test, self.true_residuals_test)

        self.l2_loss_test = tf.reduce_mean(self.autoencoder_loss_test + self.autoencoder_res_loss_test)

        dc_loss_real_test = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_randn_real,
                                                    logits=self.d_real_test))
        dc_loss_fake_test = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_randn_fake,
                                                    logits=self.d_fake_test))
        self.dc_loss_test = dc_loss_fake_test + dc_loss_real_test

        self.gen_loss_test = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_fake_test),
                                                    logits=self.d_fake))

    def use_vgg19(self):
        self.vgg19 = deeploss.VGG19Features(self.sess,
                                            feature_layers=self.config["feature_layers"],
                                            feature_weights=self.config["feature_weights"],
                                            gram_weights=self.config["gram_weights"])


    def visualize_perceptual_loss(self, x, y, out_dir, ep):
        input_x = tf.placeholder('float32', [self.config["batch_size"], self.config["spatial_size"],
                                             self.config["spatial_size"], 1],
                                           name='input_x')
        input_y = tf.placeholder('float32', [self.config["batch_size"], self.config["spatial_size"],
                                             self.config["spatial_size"], 1],
                           name='input_y')

        self.perceptual_loss = self.vgg19.make_loss_op(input_x, input_y)

        train_x_features, train_recons_features, self.percept_loss=\
            self.sess.run([self.vgg19.x_features,self.vgg19.y_features, self.perceptual_loss],
                          {input_x: x, input_y: y})

        self.vggfeatures = []

        cnt=0
        for xf, yf in zip(train_x_features, train_recons_features):
            d_train = np.abs(xf - yf)
            d_train = np.sum(d_train, axis=3, keepdims = True)
            self.vggfeatures.append(np.sum(d_train, axis=3))
            #plot_batch(np.sum(xf, axis=3, keepdims=True), self.out_dir + '/train_img_vgg_features_block'+str(cnt)+"_" + str(ep) + '.png')
            plot_batch(d_train, out_dir + '/dif_vgg_features_block'+str(cnt) + "_" + str(ep) + '.png')
            #plot_batch(d_test, self.out_dir + '/test_dif_vgg_features_block' + str(cnt) + '.png')
            #plot_batch(d_train, self.out_dir + '/dif_vgg_features_block' + str(cnt) + "_" + str(ep) + '.png')
            #plot_batch(_d_grams, self.out_dir + '/dif_vgg_grams_block' + str(cnt) + "_" + str(ep) + '.png')

            cnt += 1

    def initialize(self):
        with tf.device("/gpu:0"):
            all_variables = tf.trainable_variables()
            ae_var = [var for var in all_variables if 'dec' in var.name or 'enc' in var.name]
            ds_var = [var for var in all_variables if 'dc' in var.name]
            gen_var = [var for var in all_variables if 'enc' in var.name]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_ae = \
                    tf.train.AdamOptimizer(self.config["lr"]).minimize(self.l2_loss, var_list = ae_var)
                self.train_gen = \
                    tf.train.AdamOptimizer(self.config["lr"]).minimize(self.gen_loss, var_list = gen_var)
                self.train_dis = \
                    tf.train.AdamOptimizer(self.config["lr"]).minimize(self.dc_loss, var_list = ds_var)
        self.sess.run(tf.initializers.global_variables())

    # def summarize(self):
    #     tf.summary.scalar("train_lat_loss", tf.reduce_mean(self.latent_loss))
    #     # tf.summary.scalar("test_gen_loss", tf.reduce_mean(autoencoder_loss_test))
    #     tf.summary.scalar("test_lat_loss", tf.reduce_mean(self.latent_loss_test))
    #     self.summary_op = tf.summary.merge_all()

    def early_train(self, input_images):
        self.input_images = input_images.astype("float32")
        feed_dict = {self.image_matrix: input_images}
        self.sess.run(self.train_ae, feed_dict)
        for t in range(20):
            self.sess.run(self.train_gen, feed_dict)
            for s in range(20):
                self.sess.run(self.train_dis, feed_dict)
        self.out_mu = self.sess.run(self.decoder_output, feed_dict)
        self.residual_output = self.sess.run(self.res, feed_dict)

    def train(self, input_images):
        self.input_images = input_images.astype("float32")
        feed_dict = {self.image_matrix: input_images}
        self.sess.run(self.train_ae, feed_dict)
        self.sess.run(self.train_gen, feed_dict)
        self.sess.run(self.train_dis, feed_dict)
        self.out_mu = self.sess.run(self.decoder_output, feed_dict)
        self.residual_output = self.sess.run(self.res, feed_dict)

    def validate(self, input_images):
        self.input_images_test = input_images
        feed_dict = {self.image_matrix: self.input_images_test}

        self.out_mu_test = self.sess.run(self.decoder_output_test, feed_dict)
        self.residual_output_test = self.sess.run(self.res_test, feed_dict)

        # out_mu[input_masks==0.]=-3.5
        # out_std = sess.run(decoder_std, {image_matrix:input_images})
        # out_std = np.exp(out_std)
        #self.residual_output = self.sess.run(self.res, feed_dict)

    def visualize(self, model_name, ep):
        feed_dict = {self.image_matrix: self.input_images}
        # self.out_mu = self.sess.run(self.decoder_output, feed_dict)
        # # out_mu[input_masks==0.]=-3.5
        # # out_std = sess.run(decoder_std, {image_matrix:input_images})
        # # out_std = np.exp(out_std)
        # self.residual_output = self.sess.run(self.res, feed_dict)
        if not os.path.exists('Results/'+ model_name + '_samples/'):
            os.makedirs('Results/'+ model_name + '_samples/')
        model_name = 'Results/' + model_name
        plot_batch(self.input_images, model_name + '_samples/gr_' + str(ep) + '.png')
        plot_batch(self.out_mu, model_name + '_samples/gn_mu_' + str(ep) + '.png')
        # plot_batch(decoded_embedding[:, :, :, np.newaxis].astype("uint8"),
        #            os.path.join(wd, model_name + "_samples/embed_" + str(ep) + ".png"))
        plot_batch(self.residual_output, model_name + '_samples/res_' + str(ep) + '.png')
        plot_batch(np.abs(self.input_images - self.out_mu), model_name + '_samples/gtres_' + str(ep) + '.png')

        z_output = self.sess.run(self.z, {self.image_matrix: self.input_images})
        z_valid_output = self.sess.run(self.z_valid, {self.image_matrix: self.input_images_test})
        plt.hist(z_output.flatten(), normed=True, bins=100)
        plt.savefig(model_name + '_samples/gn_z_hist_' + str(ep) + '.png')
        plt.close()

        plt.hist(z_valid_output.flatten(), normed=True, bins=100)
        plt.savefig(model_name + '_samples/test_z_hist_' + str(ep) + '.png')
        plt.close()

        plot_batch(self.input_images_test, model_name + '_samples/test_gr_' + str(ep) + '.png')
        plot_batch(self.out_mu_test, model_name + '_samples/test_gn_mu_' + str(ep) + '.png')
        # plot_batch(decoded_embedding[:, :, :, np.newaxis].astype("uint8"),
        #            os.path.join(wd, model_name + "_samples/embed_" + str(ep) + ".png"))
        plot_batch(self.residual_output_test, model_name + '_samples/test_res_' + str(ep) + '.png')
        plot_batch(np.abs(self.input_images_test - self.out_mu_test), model_name + '_samples/test_gtres_' + str(ep) + '.png')

    def save(self,model_name, ep):
        if not os.path.exists(os.path.join(self.log_dir, model_name)):
            os.makedirs(os.path.join(self.log_dir, model_name))
        self.saver.save(self.sess, os.path.join(self.log_dir, model_name)+'/' + model_name + ".ckpt", global_step=ep)

    def load(self, model_name, step):
        model_folder = os.path.join(self.log_dir, model_name)
        self.saver.restore(self.sess, model_folder + '/' + model_name + ".ckpt-" + str(step))
