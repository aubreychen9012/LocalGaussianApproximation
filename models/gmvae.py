import tensorflow as tf
import numpy as np
import os
from networks.gmvae_network import q_wz_x, p_x_z, p_z_wc, p_c, gmloss
from utils.batches import plot_batch
from utils.losses import l2loss
from pdb import set_trace as bp

class GMVAEModel():
    def __init__(self, config, model_name, log_dir):
        gpu_config = tf.ConfigProto(allow_soft_placement=True)
        gpu_config.gpu_options.allow_growth = True
        #gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self.sess = tf.Session(config=gpu_config)
        #self.model = model()
        #self.model.__init__(model_name, image_size = config["spatial_size"])
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
        dim_c = 6  # number of clusters
        self.dim_c = dim_c
        #dim_z = 1  # latent variable z's channel
        dim_w = 1  # dimensi55on of prior w's channel
        dim_z = 1  # latent variable z's channel
          # dimension of image in vectors
        c_lambda = 20.  # the constant  on kl[p(c|w,z)||p(c)]
        clipstd = [0.0, 1.0]  # bound of std
        start_step = 0  # default starting step
        training_step = 400 * 4000  # how many steps to train at a time
        #train_rate = 5e-5


        self.image_matrix = tf.placeholder('float32',
                                           [self.config["batch_size"], self.config["spatial_size"],
                                            self.config["spatial_size"], 1],
                                           name='input')

        qwz_x_kernels = np.tile([3, 3, 3, 3, 3, 3, 1], [2, 1])  #
        qwz_x_channels = [64, 64, 64, 64, 64, 64]

        # encoding network q(z|x) and q(w|x)

        w_sampled, w_mean, w_logvar, z_sampled, z_mean, z_logvar = \
            q_wz_x(self.image_matrix, reuse=False)

        self.w_logvar = w_logvar

        # w_sampled_v, w_mean_v, w_logvar_v, z_sampled_v, z_mean_v, z_logvar_v, \
        # self.res_conv_v = \
        #     q_wz_x(self.image_matrix, dim_z, dim_w, dim_c,
        #            qwz_x_kernels[0], qwz_x_kernels[1],
        #            qwz_x_channels)

        p_z_wc_kernels = (np.ones([2, 2])).astype(int)
        p_z_wc_channels = [64, dim_z * dim_c]

        z_c_sampled, z_c_mean, z_c_logvarinv = \
            p_z_wc(w_sampled, dim_c, dim_z,
                   p_z_wc_kernels[0], p_z_wc_kernels[1],
                   p_z_wc_channels)

        # z_c_sampled_v, z_c_mean_v, z_c_logvarinv_v = \
        #     p_z_wc(w_sampled_v, dim_c, dim_z,
        #            p_z_wc_kernels[0], p_z_wc_kernels[1],
        #            p_z_wc_channels)

        # decoding network p(x|z) parameter
        px_z_kernels = np.tile([1, 3, 3, 3, 3, 3, 3], [2, 1])  #
        px_z_channels = [64, 64, 64, 64, 64, 64, 1]

        # decoding network p(x|z)

        self.decoder_output = p_x_z(z_sampled, clipstd = clipstd)
        #self.decoder_output_test = p_x_z(z_sampled_v, px_z_kernels[0], px_z_kernels[1], px_z_channels)
        # prior p(c) network
        # xz_mean, xz_logvarinv = p_x_z(z_sampled, px_z_kernels[0], px_z_kernels[1], px_z_channels, clipstd=clipstd)
        #
        # # prior p(c) network
        #
        # pc_logit, pc = p_c(z_sampled, z_c_mean, z_c_logvarinv, dim_c)
        #

        pc_logit, pc = p_c(z_sampled, z_c_mean, z_c_logvarinv, dim_c)
        self.pc = pc
        #pc_logit_v, pc_v = p_c(z_sampled_v, z_c_mean_v, z_c_logvarinv_v, dim_c)

        true_res = tf.abs(self.image_matrix-self.decoder_output)
        #true_res_test = tf.abs(self.image_matrix - self.decoder_output_test)


        #self.l2_res_loss = l2loss(true_res, self.res)
        #self.l2_res_loss_v = l2loss(true_res_test, self.res_v)


        self.t_loss, self.autoencoder_loss, self.con_loss, self.w_loss, self.c_loss = \
            gmloss(z_mean, z_logvar, z_c_mean, z_c_logvarinv, w_mean, w_logvar,
                   dim_c, pc, pc_logit, self.decoder_output, self.image_matrix,
                   z_sampled, clambda=c_lambda)

        # self.t_loss_v, self.autoencoder_loss_v, self.con_loss_v, self.w_loss_v, self.c_loss_v = \
        #     gmloss(z_mean_v, z_logvar_v, z_c_mean_v, z_c_logvarinv_v, w_mean_v, w_logvar_v,
        #            dim_c, pc_v, pc_logit_v, self.decoder_output_test, self.image_matrix,
        #            z_sampled_v, clambda=c_lambda)

        #self.t_loss += self.l2_res_loss
        #self.t_loss_v += self.l2_res_loss_v

        #self.mean_res_loss = \
        #    tf.reduce_mean(self.l2_res_loss)
        # self.mean_res_loss_v = \
        #     tf.reduce_mean(self.l2_res_loss_v)

        # self.t_loss += self.mean_res_loss
        # self.t_loss_v += self.mean_res_loss_v


    def initialize(self):
        with tf.device("/gpu:0"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(self.config["lr"]).minimize(self.t_loss)
        self.sess.run(tf.initializers.global_variables())

    def train(self, input_images):
        self.input_images = input_images.astype("float32")
        #input_masks = input_masks.astype("float32")
        feed_dict = {self.image_matrix: input_images}
        self.sess.run(self.train_op, feed_dict)
        self.out_mu, c_train = self.sess.run([self.decoder_output, self.pc], feed_dict)
        #print(c_train)
        #self.out_std = self.sess.run(self.decoder_output_std, feed_dict)
        #self.residual_output = self.sess.run(self.res, feed_dict)

    def validate(self, input_images):
        self.input_images_test = input_images
        feed_dict = {self.image_matrix: self.input_images_test}

        self.out_mu_test, c_test = self.sess.run([self.decoder_output, self.pc], feed_dict)
        print(c_test.max())
        #self.out_std_test = self.sess.run(self.decoder_output_std, feed_dict)
        #self.residual_output_test = self.sess.run(self.res, feed_dict)

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
        #plot_batch(self.out_std, model_name + '_samples/gn_std_' + str(ep) + '.png')
        # plot_batch(decoded_embedding[:, :, :, np.newaxis].astype("uint8"),
        #            os.path.join(wd, model_name + "_samples/embed_" + str(ep) + ".png"))
        #plot_batch(self.residual_output, model_name + '_samples/res_' + str(ep) + '.png')
        plot_batch(np.abs(self.input_images - self.out_mu), model_name + '_samples/gtres_' + str(ep) + '.png')

        plot_batch(self.input_images_test, model_name + '_samples/test_gr_' + str(ep) + '.png')
        plot_batch(self.out_mu_test, model_name + '_samples/test_gn_mu_' + str(ep) + '.png')
        #plot_batch(self.out_std_test, model_name + '_samples/test_gn_std_' + str(ep) + '.png')
        # plot_batch(decoded_embedding[:, :, :, np.newaxis].astype("uint8"),
        #            os.path.join(wd, model_name + "_samples/embed_" + str(ep) + ".png"))
        #plot_batch(self.residual_output_test, model_name + '_samples/test_res_' + str(ep) + '.png')
        plot_batch(np.abs(self.input_images_test - self.out_mu_test), model_name + '_samples/test_gtres_' + str(ep) + '.png')

    def save(self,model_name, ep):
        if not os.path.exists(os.path.join(self.log_dir, model_name)):
            os.makedirs(os.path.join(self.log_dir, model_name))
        self.saver.save(self.sess, os.path.join(self.log_dir, model_name)+'/' + model_name + ".ckpt", global_step=ep)

    def load(self, model_name, step):
        model_folder = os.path.join(self.log_dir, model_name)
        self.saver.restore(self.sess, model_folder + '/' + model_name + ".ckpt-" + str(step))


