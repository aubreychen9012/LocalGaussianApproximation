import os
import tensorflow as tf
import numpy as np
from utils import losses
from utils import deeploss
from utils.batches import plot_batch
from pdb import set_trace as bp


class covVAEModel():
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
        self.z_mean, self.z_A_T, self.res = \
            self.model.encoder(self.image_matrix, is_train=True, reuse=False)
        #z_stddev = tf.matmul(z_A, tf.transpose(z_A))
        samples = tf.random_normal(tf.shape(self.z_mean), 0., 1., dtype=tf.float32)
        self.guessed_z = self.z_mean + tf.matmul(samples, self.z_A_T)

        #guessed_z = z_mean + (z_A_T * samples)
        self.decoder_output, self.dec_embed = \
            self.model.decoder(self.guessed_z, name="img", is_train=True, reuse=False)

        z_mean_valid, z_A_T_valid, self.res_test = \
            self.model.encoder(self.image_matrix, is_train=False, reuse=True)
        #z_stddev_valid = tf.matmul(z_A_valid, tf.transpose(z_A_valid))
        samples_valid = tf.random_normal(tf.shape(z_mean_valid), 0., 1., dtype=tf.float32)
        guessed_z_valid = z_mean_valid + tf.matmul(samples_valid, z_A_T_valid)
        self.decoder_output_test, self.dec_embed_test = \
            self.model.decoder(guessed_z_valid, name="img", is_train=False, reuse=True)

        # summed = tf.reduce_sum(tf.square(self.decoder_output - self.image_matrix),
        #                        [1, 2, 3])
        # sqrt_summed = tf.sqrt(summed + 1e-10)
        # self.autoencoder_loss = sqrt_summed

        self.autoencoder_loss = 5.*losses.l2loss(self.decoder_output, self.image_matrix)

        #self.deep_ae_loss = 5* losses.l2loss(self.hidden_enc, self.hidden_dec)

        # self.deep_ae_loss_s2 = losses.l2loss(self.model.enc_res1,self.model.dec_res1)
        # self.deep_ae_loss_s4 = losses.l2loss(self.model.enc_res2, self.model.dec_res2)
        # self.deep_ae_loss_s8 = losses.l2loss(self.model.enc_res3, self.model.dec_res3)
        # self.deep_ae_loss_s16 = losses.l2loss(self.model.enc_res4,self.model.dec_res4)
        # self.deep_ae_loss_s1 =losses.l2loss(self.model.enc_0, self.model.dec_0)
        #
        # self.deep_ae_loss = 5*(self.deep_ae_loss_s1+self.deep_ae_loss_s2 + self.deep_ae_loss_s4 + \
        #                     self.deep_ae_loss_s8+self.deep_ae_loss_s16)

        self.true_residuals = tf.abs(self.image_matrix - self.decoder_output)
        self.autoencoder_res_loss = losses.l2loss(self.res, self.true_residuals)

        #å
        # residuals = np.abs(self.decoder_output - self.image_matrix)
        # summed = tf.reduce_sum(tf.square(self.res - residuals), axis=[1, 2, 3])
        # sqrt_summed = tf.sqrt(summed + 1e-10)
        # self.autoencoder_res_loss = sqrt_summed

        # 1d KL divergence
        #self.latent_loss = tf.reduce_sum(
        #  tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev) + 1e-10) - 1, 1)

        # nd KL
        self.latent_loss = losses.kl_cov_gaussian(self.z_mean, self.z_A_T)


        self.loss = tf.reduce_mean(self.autoencoder_loss +
                                   self.latent_loss + self.autoencoder_res_loss)

        ## validate
        self.autoencoder_loss_test = 5.*losses.l2loss(self.decoder_output_test, self.image_matrix)
        #self.deep_ae_loss_test = 5. * losses.l2loss(self.hidden_enc_test, self.hidden_dec_test)
        self.true_residuals_test = tf.abs(self.image_matrix - self.decoder_output_test)
        self.autoencoder_res_loss_test = losses.l2loss(self.res_test, self.true_residuals_test)

        self.latent_loss_test = losses.kl_cov_gaussian(z_mean_valid, z_A_T_valid)

        self.loss_test = tf.reduce_mean(self.autoencoder_loss_test +
                                        self.latent_loss + self.autoencoder_res_loss_test)

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

        # test_x_features, test_recons_features = \
        #     self.sess.run([self.vgg19.x_features, self.vgg19.y_features])

        self.vggfeatures = []
        # self.d_test_features = []
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
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.config["lr"]).minimize(self.loss)
        self.sess.run(tf.initializers.global_variables())

    def summarize(self):
        tf.summary.scalar("train_lat_loss", tf.reduce_mean(self.latent_loss))
        # tf.summary.scalar("test_gen_loss", tf.reduce_mean(autoencoder_loss_test))
        tf.summary.scalar("test_lat_loss", tf.reduce_mean(self.latent_loss_test))
        self.summary_op = tf.summary.merge_all()

    def train(self, input_images):
        #input_images, input_masks = next(batches)[:2]
        self.input_images = input_images.astype("float32")
        #input_masks = input_masks.astype("float32")
        feed_dict = {self.image_matrix: input_images}
        self.sess.run(self.train_op, feed_dict)
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
