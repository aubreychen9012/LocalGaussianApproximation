import os
import tensorflow as tf
import numpy as np
from utils import losses
from utils.batches import plot_batch
from pdb import set_trace as bp
from utils import deeploss


class VAEPerceptualModel():
    def __init__(self, model, config, model_name, log_dir):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.image_matrix = tf.placeholder('float32',
                                           [config["batch_size"], config["spatial_size"], config["spatial_size"], 1],
                                           name='input')
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
        self.out_dir = 0
        self.input_image = 0
        #self.decoder_output = 0


    def build_network(self):
        self.z_mean, self.z_std, self.res = self.model.encoder(self.image_matrix, is_train=True, reuse=False)
        #z_stddev = tf.matmul(z_A, tf.transpose(z_A))
        samples = tf.random_normal(tf.shape(self.z_mean), 0., 1., dtype=tf.float32)
        guessed_z = self.z_mean + self.z_std*samples

        #guessed_z = z_mean + (z_A_T * samples)
        self.decoder_output, self.dec_embed = self.model.decoder(guessed_z, name="img", is_train=True, reuse=False)

        z_mean_valid, z_std_valid, self.res_test = self.model.encoder(self.image_matrix, is_train=False, reuse=True)
        #z_stddev_valid = tf.matmul(z_A_valid, tf.transpose(z_A_valid))
        samples_valid = tf.random_normal(tf.shape(z_mean_valid), 0., 1., dtype=tf.float32)
        guessed_z_valid = z_mean_valid + z_std_valid*samples_valid
        self.decoder_output_test, self.dec_embed_test = self.model.decoder(guessed_z_valid, name="img", is_train=False, reuse=True)

        self.vgg19 = deeploss.VGG19Features(self.sess,
                                             feature_layers=self.config["feature_layers"],
                                             feature_weights=self.config["feature_weights"],
                                             gram_weights=self.config["gram_weights"])

        # summed = tf.reduce_sum(tf.square(self.decoder_output - self.image_matrix),
        #                        [1, 2, 3])
        # sqrt_summed = tf.sqrt(summed + 1e-10)
        # self.autoencoder_loss = sqrt_summed

        #self.autoencoder_loss = losses.l2loss(self.decoder_output, self.image_matrix)
        self.true_residuals = tf.abs(self.image_matrix-self.decoder_output)
        self.autoencoder_res_loss = losses.l2loss(self.res, self.true_residuals)
        self.autoencoder_loss = 5.0 * self.vgg19.make_loss_op(self.image_matrix,self.decoder_output)


        #
        # residuals = np.abs(self.decoder_output - self.image_matrix)
        # summed = tf.reduce_sum(tf.square(self.res - residuals), axis=[1, 2, 3])
        # sqrt_summed = tf.sqrt(summed + 1e-10)
        # self.autoencoder_res_loss = sqrt_summed

        # 1d KL divergence
        self.latent_loss = losses.kl_loss_1d(self.z_mean, self.z_std)

        # nd KL
        #self.latent_loss = losses.kl_cov_gaussian(self.z_mean, self.z_A_T)


        self.loss = tf.reduce_mean(self.autoencoder_loss) + 1.*tf.maximum(tf.reduce_mean(self.latent_loss), 100.) \
                    + tf.reduce_mean(self.autoencoder_res_loss)

        ## validate
        #self.autoencoder_loss_test = losses.l2loss(self.decoder_output_test, self.image_matrix)
        self.autoencoder_loss_test = \
            5.0 * self.vgg19.make_loss_op(self.image_matrix, self.decoder_output_test)

        self.true_residuals_test = tf.abs(self.image_matrix - self.decoder_output_test)
        self.autoencoder_res_loss_test = losses.l2loss(self.res_test, self.true_residuals_test)

        self.latent_loss_test = losses.kl_loss_1d(z_mean_valid, z_std_valid)

        self.loss_test = tf.reduce_mean(self.autoencoder_loss_test) + 1.*tf.maximum(tf.reduce_mean(self.latent_loss_test),100.)\
                         + tf.reduce_mean(self.autoencoder_res_loss_test)

    def initialize(self):
        with tf.device("/gpu:0"):
            #opt_op = optimizer.minimize(loss, var_list=self.trainable_variables)
            self.trainable_variables = [v for v in tf.trainable_variables()
                                        if not v in self.vgg19.variables]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.config["lr"]).\
                    minimize(self.loss, var_list = self.trainable_variables)
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

    def validate(self, input_images):
        self.input_images_test = input_images.astype("float32")
        feed_dict = {self.image_matrix: self.input_images_test}

        self.out_mu_test = self.sess.run(self.decoder_output_test, feed_dict)
        self.out_mu_test = self.out_mu_test.astype("float32")
        self.residual_output_test = self.sess.run(self.res_test, feed_dict)

    def visualize(self, model_name, ep):
        feed_dict = {self.image_matrix: self.input_images}
        self.out_mu = self.sess.run(self.decoder_output, feed_dict)
        self.out_mu = self.out_mu.astype("float32")
        # out_mu[input_masks==0.]=-3.5
        # out_std = sess.run(decoder_std, {image_matrix:input_images})
        # out_std = np.exp(out_std)
        self.residual_output = self.sess.run(self.res, feed_dict)
        if not os.path.exists('Results/'+ model_name + '_samples/'+str(ep)):
            os.makedirs('Results/'+ model_name + '_samples/' + str(ep))
        self.out_dir = 'Results/' + model_name + '_samples/' + str(ep)
        plot_batch(self.input_images, self.out_dir+ '/gr_' + str(ep) + '.png')
        plot_batch(self.out_mu, self.out_dir + '/gn_mu_' + str(ep) + '.png')
        # plot_batch(decoded_embedding[:, :, :, np.newaxis].astype("uint8"),
        #            os.path.join(wd, model_name + "_samples/embed_" + str(ep) + ".png"))
        plot_batch(self.residual_output, self.out_dir + '/res_' + str(ep) + '.png')
        plot_batch(np.abs(self.input_images - self.out_mu),
                   self.out_dir + '/gtres_' + str(ep) + '.png')

        plot_batch(self.input_images_test, self.out_dir + '/test_gr_' + str(ep) + '.png')
        plot_batch(self.out_mu_test, self.out_dir + '/test_gn_mu_' + str(ep) + '.png')
        # plot_batch(decoded_embedding[:, :, :, np.newaxis].astype("uint8"),
        #            os.path.join(wd, model_name + "_samples/embed_" + str(ep) + ".png"))
        plot_batch(self.residual_output_test, self.out_dir + '/test_res_' + str(ep) + '.png')
        plot_batch(np.abs(self.input_images_test - self.out_mu_test),
                   self.out_dir + '/test_gtres_' + str(ep) + '.png')

    def visualize_vgg_features(self, ep, out_dir=None, training=1):
        # if not self.input_image:
        #     training = 0
        if out_dir:
            self.out_dir = out_dir

        cnt = 1

        if training:
            train_x_features, train_recons_features, train_x_grams, train_recons_grams =\
                self.sess.run([self.vgg19.x_features,self.vgg19.y_features,
                               self.vgg19.x_grams, self.vgg19.y_grams],
                          {self.image_matrix: self.input_images,self.decoder_output: self.out_mu})

        # _ = self.vgg19.make_loss_op(self.input_images, self.out_mu)
        # _ = self.sess.run(self.autoencoder_loss, )
        # train_x_features = self.vgg19.x_features
        # train_recons_features = self.vgg19.y_features

        test_x_features, test_recons_features, test_x_grams, test_recons_grams = \
            self.sess.run([self.vgg19.x_features, self.vgg19.y_features,
                           self.vgg19.x_grams, self.vgg19.y_grams],
                          {self.image_matrix: self.input_images_test,
                           self.decoder_output_test: self.out_mu_test})

        # _ = self.vgg19.make_loss_op(self.input_images_test, self.out_mu_test)
        # test_x_features = self.vgg19.x_features
        # test_recons_features = self.vgg19.y_features
        self.d_train_features = []
        self.d_test_features = []
        if training:
            for xf, yf, xf_test, yf_test in zip(train_x_features, train_recons_features,
                                                test_x_features, test_recons_features):
                d_train = np.abs(xf - yf)
                d_test = np.abs(xf_test - yf_test)
                #d_grams = np.abs(xg-yg)

                #_d_grams = np.sum(d_grams, axis=3, keepdims=True)

                d_train = np.sum(d_train, axis=3, keepdims = True)
                self.d_train_features.append(np.sum(d_train, axis=3))
                d_test = np.sum(d_test, axis=3, keepdims = True)
                self.d_test_features.append(np.sum(d_test, axis=3))

                plot_batch(np.sum(xf, axis=3, keepdims=True), self.out_dir + '/train_img_vgg_features_block'+str(cnt)+"_" + str(ep) + '.png')
                plot_batch(d_train, self.out_dir + '/dif_vgg_features_block'+str(cnt)+"_" + str(ep) + '.png')
                plot_batch(d_test, self.out_dir + '/test_dif_vgg_features_block' + str(cnt) + "_" + str(ep) + '.png')
                plot_batch(d_train, self.out_dir + '/dif_vgg_features_block' + str(cnt) + "_" + str(ep) + '.png')
                #plot_batch(_d_grams, self.out_dir + '/dif_vgg_grams_block' + str(cnt) + "_" + str(ep) + '.png')

                cnt += 1
            #self.d_train_features = np.abs(train_x_features - train_recons_features)
            #self.d_test_features = np.abs(test_x_features-test_recons_features)
            #self.d_train_features = np.sum(self.d_train_features, axis=-1)
            #self.d_test_features = np.sum(self.d_test_features, axis=-1)
        elif not training:
            for xf_test, yf_test in zip(test_x_features, test_recons_features):
                #d_train = np.abs(xf - yf)
                d_test = np.abs(xf_test - yf_test)
                #d_train = np.sum(d_train, axis=3, keepdims = True)
                d_test = np.sum(d_test, axis=3, keepdims = True)
                self.d_test_features.append(np.sum(d_test, axis=3))
                #plot_batch(np.sum(xf, axis=3, keepdims=True), self.out_dir + '/train_img_vgg_features_block'+str(cnt)+"_" + str(ep) + '.png')
                #plot_batch(d_train, self.out_dir + '/dif_vgg_features_block'+str(cnt)+"_" + str(ep) + '.png')
                plot_batch(d_test, self.out_dir + '/test_dif_vgg_features_block' + str(cnt) + "_" + str(ep) + '.png')
                cnt += 1
            #self.d_test_features = np.abs(test_x_features - test_recons_features)
            #self.d_test_features = np.sum(self.d_test_features, axis=-1)



    def save(self,model_name, ep):
        if not os.path.exists(os.path.join(self.log_dir, model_name)):
            os.makedirs(os.path.join(self.log_dir, model_name))
        self.saver.save(self.sess, os.path.join(self.log_dir, model_name)+'/' + model_name + ".ckpt", global_step=ep)

    def load(self, model_name, step):
        model_folder = os.path.join(self.log_dir, model_name)
        self.saver.restore(self.sess, model_folder + '/' + model_name + ".ckpt-" + str(step))
