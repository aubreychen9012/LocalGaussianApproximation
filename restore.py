import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SGE_GPU']
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import tensorflow as tf
import numpy as np
import time
import scipy
import nibabel as nib
from scipy import interpolate
import yaml
import timeit
import argparse
from networks.vae_res_bilinear_conv import VariationalAutoencoder

from models.vae import VAEModel
from utils import losses
from utils.batches import get_camcan_batches,tile, plot_batch
from utils.rank_one import RankOne, rank_one_np, rank_one_update_np
from utils import threshold
from pdb import set_trace as bp
from restore_camcan import restore_camcan


from preprocess.preprocess import *
from metrics.auc_score import compute_tpr_fpr
from metrics.dice import dsc, dsc_compute
from sklearn.metrics import roc_auc_score


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v,f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

class Restoration():
    def __init__(self, input_images, mask_images, prior_model, model_name, config):
        tf.reset_default_graph()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.model = prior_model(model_name, image_size=config["spatial_size"])
        self.config = config
        self.input_image = input_images
        self.mask_image = mask_images

    def reconstructor(self):
        init_values = tf.constant_initializer(self.input_image)
        init_masks = tf.constant_initializer(self.mask_image)
        self.img_ano = tf.get_variable('img_ano', [self.config["batch_size"], self.config["spatial_size"],
                                                   self.config["spatial_size"], 1], initializer=init_values)
        self.img_mask = tf.get_variable('img_mask', [self.config["batch_size"], self.config["spatial_size"],
                                                     self.config["spatial_size"], 1], initializer=init_masks)
        self.res = tf.placeholder('float32', [self.config["batch_size"], self.config["spatial_size"],
                                              self.config["spatial_size"], 1],
                                  name='residuals')
        self.res_mu = tf.placeholder('float32', [self.config["batch_size"], self.config["spatial_size"],
                                                 self.config["spatial_size"], 1],
                                     name='residuals_mu')

        self.z_mean, self.z_A_T, self.pred_res = \
            self.model.encoder(self.img_ano, is_train=True, reuse=False)

        samples = tf.random_normal(tf.shape(self.z_mean), 0., 1., dtype=tf.float32)
        self.guessed_z = self.z_mean + tf.matmul(samples, self.z_A_T)

        self.decoder_output, _ = \
            self.model.decoder(self.guessed_z, name='img', is_train=True, reuse=False)
        self.autoencoder_loss = losses.l1loss(self.decoder_output, self.img_ano)

        # self.residuals = tf.abs(self.img_ano - self.decoder_output)
        self.autoencoder_res_loss = losses.l1loss(self.res, self.pred_res)
        weighted_l2 = (self.res_mu - self.img_ano) ** 2 / (self.res ** 2 + 1e-10)
        clip_value = tf.contrib.distributions.percentile(weighted_l2, q=95)

        weighted_l2 = tf.clip_by_value(weighted_l2, -5e3, 5e3)
        self.w = (self.res_mu - self.img_ano) ** 2 / (self.res ** 2 + 1e-10)
        self.autoencoder_l2_weight = tf.reduce_sum(weighted_l2,
                                                   axis=[1, 2, 3])
        self.loss_vae = tf.reduce_mean(self.autoencoder_loss)  # + self.loss_m)
        self.loss_residual = tf.reduce_mean(self.autoencoder_res_loss)

        t_vars = tf.trainable_variables()
        load_vars = [var for var in t_vars if "encoder" in var.name or "decoder" in var.name]
        self.load_vars = load_vars
        self.saver = tf.train.Saver(load_vars)

    def reconstructor_vae(self):
        init_values = tf.constant_initializer(self.input_image)
        init_masks = tf.constant_initializer(self.mask_image)
        self.img_ano = tf.get_variable('img_ano', [self.config["batch_size"], self.config["spatial_size"],
                                            self.config["spatial_size"], 1], initializer=init_values)
        self.img_mask = tf.get_variable('img_mask', [self.config["batch_size"], self.config["spatial_size"],
                                            self.config["spatial_size"], 1], initializer=init_masks)
        self.res = tf.placeholder('float32', [self.config["batch_size"], self.config["spatial_size"],
                                            self.config["spatial_size"], 1],
                                           name='residuals')
        self.res_mu = tf.placeholder('float32', [self.config["batch_size"], self.config["spatial_size"],
                                              self.config["spatial_size"], 1],
                                  name='residuals_mu')

        self.z_mean, self.z_std, self.pred_res = \
                self.model.encoder(self.img_ano, is_train=True, reuse=False)
        self.z_std = tf.exp(self.z_std)
        # z_stddev = tf.matmul(z_A, tf.transpose(z_A))
        samples = tf.random_normal(tf.shape(self.z_mean), 0., 1., dtype=tf.float32)
        self.guessed_z = self.z_mean + self.z_std*samples #
        self.decoder_output = \
            self.model.decoder(self.guessed_z, name="img", is_train=True, reuse=False)
        self.autoencoder_loss_l2= (self.decoder_output-self.img_ano)**2

        #self.autoencoder_loss = losses.gaussian_negative_log_likelihood(self.decoder_output,
        #                                                                self.img_ano,
        #                                                                tf.exp(1.4))

        #self.residuals = tf.abs(self.img_ano - self.decoder_output)
        self.autoencoder_res_loss = losses.l1loss(self.res, self.pred_res)

        self.autoencoder_loss = losses.gaussian_negative_log_likelihood(self.decoder_output,
                                                                        self.img_ano,
                                                                        tf.exp(1.4))
        weighted_l2 = (self.res_mu-self.img_ano)**2/(self.res**2+1e-10)
        clip_value = tf.contrib.distributions.percentile(weighted_l2, q=95)

        weighted_l2 = tf.clip_by_value(weighted_l2, -5e3, 5e3)
        self.w = (self.res_mu - self.img_ano) ** 2 / (self.res**2+1e-10)
        self.autoencoder_l2_weight = tf.reduce_sum(weighted_l2,
                                                   axis=[1,2,3])
        #self.loss_m = 10.*tf.reduce_sum(tf.abs(self.img_mask), axis=[1,2,3])

        # nd KL
        self.latent_loss = losses.kl_loss_1d(self.z_mean, self.z_std)
        #self.latent_loss = losses.kl_cov_gaussian(self.z_mean, self.z_std)
        #self.loss = tf.reduce_mean(self.autoencoder_loss + self.latent_loss
        #                           + self.autoencoder_res_loss + self.loss_m)

        self.loss_vae = tf.reduce_mean(self.autoencoder_loss + self.latent_loss) # + self.loss_m)
        self.loss_residual = tf.reduce_mean(self.autoencoder_res_loss)

        t_vars = tf.trainable_variables()
        load_vars = [var for var in t_vars if "encoder" in var.name or "decoder" in var.name]
        self.saver = tf.train.Saver(load_vars)

    def reconstructor_z4094(self):
        init_values = tf.constant_initializer(self.input_image)
        init_masks = tf.constant_initializer(self.mask_image)
        self.img_ano = tf.get_variable('img_ano', [self.config["batch_size"], self.config["spatial_size"],
                                                   self.config["spatial_size"], 1], initializer=init_values)
        self.img_mask = tf.get_variable('img_mask', [self.config["batch_size"], self.config["spatial_size"],
                                                     self.config["spatial_size"], 1], initializer=init_masks)
        self.res = tf.placeholder('float32', [self.config["batch_size"], self.config["spatial_size"],
                                              self.config["spatial_size"], 1],
                                  name='residuals')
        self.res_mu = tf.placeholder('float32', [self.config["batch_size"], self.config["spatial_size"],
                                                 self.config["spatial_size"], 1],
                                     name='residuals_mu')

        self.z_mean, self.z_std, self.pred_res = self.model.encoder(self.img_ano,
                                                               is_train=True, reuse=False)
        # z_stddev = tf.matmul(z_A, tf.transpose(z_A))
        samples = tf.random_normal(tf.shape(self.z_mean), 0., 1., dtype=tf.float32)
        self.guessed_z = self.z_mean + self.z_std * samples

        # guessed_z = z_mean + (z_A_T * samples)
        self.decoder_output = self.model.decoder(self.guessed_z, name="img",
                                                 is_train=True, reuse=False)

        z_mean_valid, z_std_valid, self.res_test = self.model.encoder(self.image_matrix,
                                                                      is_train=False, reuse=True)
        # z_stddev_valid = tf.matmul(z_A_valid, tf.transpose(z_A_valid))
        samples_valid = tf.random_normal(tf.shape(z_mean_valid), 0., 1., dtype=tf.float32)
        guessed_z_valid = z_mean_valid + z_std_valid * samples_valid
        self.decoder_output_test = self.model.decoder(guessed_z_valid,
                                                      name="img", is_train=False, reuse=True)
        #self.autoencoder_loss = losses.l1loss(self.decoder_output, self.img_ano)

        # self.residuals = tf.abs(self.img_ano - self.decoder_output)
        self.autoencoder_res_loss = losses.l1loss(self.res, self.pred_res)

        # mu = tf.reshape(self.res_mu - self.img_ano, [batch_size, -1])
        # mu = tf.linalg.transpose(mu)
        #
        # weighted_l2 = tf.multiply(mu, tf.linalg.inv(self.pred_res))
        # weighted_l2 = tf.multiply(weighted_l2, tf.linalg.tranpose(mu))

        weighted_l2 = (self.res_mu - self.img_ano) ** 2 / (self.res ** 2 + 1e-10)

        # x_flatten = tf.linalg.transpose(tf.reshape(self.res_mu-self.img_ano, [-1]))
        # x_flatten_T = tf.linalg.transpose(x_flatten)
        #
        # weighted_l2_part1 = tf.matmul(tf.matmul(x_flatten,tf.linalg.inv(self.pred_res)),
        #                               x_flatten_T)
        # mu_matrix = tf.matmul(x_flatten, x_flatten_T)
        # weighted_l2_part2 = tf.matmul(tf.matmul(x_flatten, tf.linalg.inv(mu_matrix)),
        #                               x_flatten_T)
        # weighted_l2 = weighted_l2_part1 + weighted_l2_part2

        #weighted_l2 = tf.clip_by_value(weighted_l2, -5e3, 5e3)
        #self.w = (self.res_mu - self.img_ano) ** 2 / (self.res ** 2 + 1e-10)
        self.autoencoder_l2_weight = tf.reduce_sum(weighted_l2,
                                                   axis=[1, 2, 3])

        # nd KL
        self.latent_loss = losses.kl_loss_1d(self.z_mean, self.z_std)

        self.loss_vae = tf.reduce_mean(self.autoencoder_loss + self.latent_loss)  # + self.loss_m)
        self.loss_residual = tf.reduce_mean(self.autoencoder_res_loss)

        t_vars = tf.trainable_variables()
        load_vars = [var for var in t_vars if "encoder" in var.name or "decoder" in var.name]
        self.saver = tf.train.Saver(load_vars)

    # def parameter_estimator(self, z_samples):
    #     dist = tf.contrib.distributions.Normal([.0]*z_samples.shape[-1], [1.]*z_samples.shape[-1])
    #     z_tensor = tf.variable(z, name="z_tensor")
    #     z_prob = tf.map_fn(dist.prob, z_tensor)
    #
    #     x_flatten = tf.linalg.transpose(tf.reshape(self.res_mu - self.img_ano, [-1]))
    #     x_flatten_T = tf.linalg.transpose(x_flatten)
    #
    #     weighted_l2_part1 = tf.matmul(tf.matmul(x_flatten, tf.linalg.inv(self.pred_res)),
    #                                   x_flatten_T)
    #     mu_matrix = tf.matmul(x_flatten, x_flatten_T)
    #     weighted_l2_part2 = tf.matmul(tf.matmul(x_flatten, tf.linalg.inv(mu_matrix)),
    #                                   x_flatten_T)
    #     weighted_l2 = weighted_l2_part1 + weighted_l2_part2
    #
    #     self.w = (self.res_mu - self.img_ano) ** 2 / (self.res ** 2 + 1e-10)
    #     self.autoencoder_l2_weight = tf.reduce_sum(weighted_l2,
    #                                                axis=[1, 2, 3])


    def load(self, log_dir, model_name, step):
        model_folder = os.path.join(log_dir, model_name)
        self.saver.restore(self.sess, model_folder + '/' + model_name + ".ckpt-" + str(step))

    def run(self, prior_residuals, update_lr, restore_iteration):
        riter = restore_iteration
        t_vars = tf.trainable_variables()
        update_vars = [var for var in t_vars if 'img_ano' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.gfunc = self.loss_vae + tf.image.total_variation(tf.abs(self.img_ano-self.input_image))
        with tf.control_dependencies(update_ops):
            adam = tf.train.AdamOptimizer(0.05)
            grad_vars = adam.compute_gradients(self.loss_vae, var_list = update_vars)
            grad_kl = adam.compute_gradients(self.latent_loss, var_list=load_vars)
            #img_mask = tf.constant(img_mask, dtype=tf.float32)
            #masked_gradients = grad_vars[0][0]*img_mask
            restore_op = adam.apply_gradients([(tf.clip_by_value(grad_vars[0][0]*self.img_mask,
                                                                 -5,5), grad_vars[0][1])])
            #grad_vars = adam.compute_gradients(self.loss_residual, var_list=update_vars)
            #residual_op = adam.apply_gradients([(grad_vars[0][0]*self.img_mask, grad_vars[0][1])])
            #restore_op = tf.train.AdamOptimizer(0.1).minimize(self.loss,
            #                                                        var_list=update_vars)
        # initialize_uninitialized(self.sess)
        for i in range(riter):
            loss_vae, pred_res, _ = self.sess.run([self.gfunc, self.pred_res, restore_op],
                                              {self.res: prior_residuals})
            # if i%100==0:
            #     print(loss_vae)
            #     restored_images = self.sess.run(self.img_ano)
            #     #restored_mask = self.sess.run(self.img_mask)
            #     restored_images[self.mask_image==0]=-3.5
            for j in range(3):
                loss_residual, pred_res, _ = self.sess.run([self.loss_residual, self.pred_res,
                                                            residual_op],
                                              {self.res: prior_residuals})
                # restored_images = self.sess.run(self.img_ano)
            if i % 100 == 0:
                print(loss_vae, loss_residual)
                restored_images = self.sess.run(self.img_ano)
                # restored_mask = self.sess.run(self.img_mask)
                #restored_images[self.mask_image == 0] = -3.5
                #dev = restored_images.min()+3.5
                #restored_images -=dev
                plot_batch(restored_images, vis_dir + '/_images_restored_' + str(i) + '.png')
                #plot_batch(restored_mask, vis_dir + '/_masks_restored_' + str(i) + '.png')

        return restored_images, pred_res

    def gaussian_approximator(self, weight, riter, prior_residuals, dec_mu, constraint='L1'):
        t_vars = tf.trainable_variables()
        update_vars = [var for var in t_vars if 'img_ano' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #self.gfunc = tf.reduce_mean(self.autoencoder_l2/(self.res**2+1e-10) + weight*tf.image.total_variation(
        #     tf.abs(self.input_image-self.img_ano)))
        #self.gfunc = tf.reduce_mean(self.autoencoder_l2 * tf.exp(self.res) + weight * tf.image.total_variation(
        #    tf.abs(self.input_image - self.img_ano)))

        self.gfunc = tf.reduce_mean(self.autoencoder_l2_weight)
        if "L1" in constraint:
            self.gfunc += weight * tf.reduce_mean(losses.l1loss(
                self.input_image, self.img_ano))
            if "combined" in constraint:
                self.gfunc += self.loss_residual

        # TV constraint
        elif "TV" in constraint:
            self.gfunc += \
                weight * tf.reduce_mean(tf.image.total_variation(self.img_ano - self.input_image))
            if "combined" in constraint:
                self.gfunc += self.loss_residual

        with tf.control_dependencies(update_ops):
            adam = tf.train.AdamOptimizer(0.005) # tv: 0.005  l1: 0.005  probl1:0.01
            grad_vars = adam.compute_gradients(self.gfunc, var_list=update_vars)
            self.grad_kl = tf.gradients([self.latent_loss], [self.img_ano])
            self.grad_rec = tf.gradients([self.autoencoder_loss], [self.img_ano])
            self.grad = [tf.clip_by_value(grad * self.img_mask, -5., 5.) for grad, var in grad_vars]
            restore_op = adam.apply_gradients([(grad_vars[0][0] * self.img_mask, grad_vars[0][1])])

        initialize_uninitialized(self.sess)
        img_batch0 = img_batch.copy()
        for i in range(riter):
            rec_loss, l2_loss, grad_rec, grad_kl = self.sess.run([self.autoencoder_loss,
                                                        self.autoencoder_loss_l2,
                                                                  self.grad_rec,
                                   self.grad_kl],
                                  {self.res: prior_residuals, self.res_mu: dec_mu})

            # plot_batch(rec_loss.reshape(64,128,128,1), vis_dir + '/rec_loss_' + str(i) + '.png')
            # plot_batch(grad_kl[0], vis_dir + '/grad_kl_' + str(i) + '.png')
            # plot_batch(grad_rec[0], vis_dir + '/grad_rec_' + str(i) + '.png')
            # plot_batch(l2_loss.reshape(64,128,128,1), vis_dir + '/rec_l2_loss_' + str(i) + '.png')
            if i%200==0:
                #dec_mu = self.sess.run(self.decoder_output)
                decoded_mu = np.zeros(img_batch.shape)
                decoded_concat = []
                w_im_array = []
                for s in range(10):
                    recons_samples = self.sess.run(self.decoder_output)
                    decoded_mu+=recons_samples

                dec_mu = decoded_mu/10.

            loss_vae, pred_res, _ = self.sess.run([self.loss_vae, self.pred_res, restore_op],
                                                 {self.res: prior_residuals,
                                                  self.res_mu: dec_mu})

            loss_residual, pred_res = self.sess.run([self.gfunc, self.pred_res],
                                                          {self.res: prior_residuals,
                                                           self.res_mu: dec_mu})

            if i % 100 == 0:
                print(loss_vae, loss_residual)
                grads = self.sess.run(self.grad, {self.res: prior_residuals, self.res_mu: dec_mu})
                grads = grads[0]
                grads[self.img_mask==0]=0
                w_clip = self.sess.run(self.w, {self.res: prior_residuals,
                                                self.res_mu: dec_mu})

                restored_images = self.sess.run(self.img_ano)
                plot_batch(restored_images, vis_dir + '/images_batch_restored_' + str(i) + '.png')

        restored_images = self.sess.run(self.img_ano)
        pred_res = self.sess.run(self.pred_res,
                                 {self.res: prior_residuals,
                                  self.res_mu: dec_mu})
        return restored_images, pred_res

    def inv_solver(self, x, y):
        h = tf.shape(x)[0]
        self.w = tf.Variable(tf.random_normal(shape=[1,h], mean=0., stddev=0.1), 'inv_row')
        self.w = tf.cast(w, dtype=tf.float32)
        inv_vars = [v for v in tf.global_variables() if v.name == "inv_row"][0]

        _y = tf.einsum("i,ij->ij", self.w,x)
        loss = tf.reduce_sum((y-_y)**2)
        self.op = tf.train.AdamOptimizer(1e-2).minimize(loss, var_list = inv_vars)


    def gaussian_approximator_estimate_cov(self, img_batch, weight, riter, constraint = "L1"):
        x = tf.Variable(tf.zeros([128*128,128*128],tf.float32), name="mat_tinv")
        y = tf.Variable(tf.zeros([1,128*128],tf.float32), name="inv_target")

        initialize_uninitialized(self.sess)
        print(timeit.timeit())


        if "L1" in constraint:
           pass

        elif "TV" in constraint:
            img_batch0 = img_batch.copy()
            for r in range(riter):
                decoded_mu = np.zeros(img_batch.shape)
                decoded_concat = np.zeros((n_latent_samples, batch_size,
                                          img_size, img_size,1))
                w_im_array = np.zeros((n_latent_samples, batch_size))

                z_mean, z_cov = self.sess.run([self.z_mean, self.z_A_T])
                #pred_res = self.sess.run(self.pred_res, {self.img_ano:img_batch0})
                print(timeit.timeit())

                for s in range(n_latent_samples):
                    samples = np.random.normal(0., 1., z_mean.shape)
                    #z_samples = z_mean + np.dot(samples, z_cov)
                    z_samples, recons_samples = self.sess.run([
                        self.guessed_z, self.decoder_output],
                        {self.z_mean:z_mean,
                         self.z_A_T: z_cov})
                    w_im = importance_weight(z_mean, z_cov, z_samples)
                    w_im_array[s] += w_im
                    #decoded_concat.append(recons_samples)
                    decoded_mu += np.einsum('i,ijkh->ijkh', w_im, recons_samples)
                    decoded_concat[s]+= recons_samples
                    print(s)

                w_im_array = np.array(w_im_array).astype("float32")
                decoded_concat = np.array(decoded_concat)

                # dec_mu_p2 = np.mean(decoded_concat**2, axis = 0)


                w_im_sum = np.sum(w_im_array, axis=0)
                w_im_array = np.array([w_im_array[:, i] / w_im_sum[i] for i in range(64)])
                #w_im_array = w_im_array.T
                decoded_mu = np.einsum('ijkh,i->ijkh', decoded_mu, 1 / w_im_sum)

                #dec_mu = np.mean(decoded_concat, axis=0)
                # err = err.reshape(10,64,128,128)
                #
                # decoded_std = np.std(np.transpose(decoded_concat.reshape(10,64,128,128), [1,2,3,0]), axis=3)
                # plot_batch(pred_res+dec_mu_p2-decoded_mu**2, vis_dir + '/_decoded_er_var2.png')

                # for i in range(n_latent_samples):
                #     decoded_mu += np.array([w_im_array[i, j] * decoded_concat[i, j]
                #                             for j in range(64)])

                matmul_flatten = \
                    (decoded_mu - decoded_concat).reshape(n_latent_samples, 64, -1, 1)
                matmul_flatten = matmul_flatten.astype("float32")

                #sigma_flatten = \
                #    (img_batch - decoded_concat).reshape(n_latent_samples, 64, -1, 1)
                mu_flatten = \
                    (img_batch - decoded_mu).reshape(64, -1)
                mu_flatten = mu_flatten.astype("float32")
                opt_grad = tf.zeros(64,128*128)
                #plot_batch(img_batch-decoded_mu, vis_dir + '/_dif_' + str(i) + '.png')

                # for b in range(batch_size):
                #     for n in range(n_latent_samples):
                #         _ = rank_one.sess.run(rank_one.rank_one_update(),
                #                           {rank_one.u: matmul_flatten[n,b],
                #                            rank_one.mu: mu_flatten[b],
                #                            rank_one.w: w_im_array[n,b]})
                #         print(timeit.timeit())
                #     grad_n = rank_one.sess.run(self.grad,
                #                            {rank_one.u: matmul_flatten[n, b].astype("float32"),
                #                             rank_one.mu: mu_flatten[b].astype("float32"),
                #                             rank_one.w: w_im_array[n, b]}
                #                            )
                #     rank_one.step_one()
                #     grad[n]+=grad_n

                #A_inv = np.eye(128*128)
                vector_ones = tf.ones(128*128)
                print(timeit.timeit())
                for b in range(64):
                    for i in range(128*128):
                        for _ in range(10):
                            _, w = self.sess.run([self.op, self.w], {x: matmul_flatten[b,i],
                                                      y: vector_ones[i]+1})
                            bp()

                        opt_grad[b,i] += tf.einsum("ij,ji->ii", w, mu_flatten[b])[0][0]


                # for b in range(batch_size):
                #     # cov_inv = self.sess.run(rank_one.M_i,
                #     #                         {rank_one.u: matmul_flatten[:,b],
                #     #                          rank_one.w: w_im_array[b,:]}
                #     #                         )
                #     cov_inv = rank_one_update_np(matmul_flatten[:,b],
                #                           w_im_array[b,:],
                #                           batch_size,
                #                           n_latent_samples)
                #     grad_b = np.einsum("ij,i->i", cov_inv, mu_flatten[b])
                #     grad_b = grad_b.reshape(128,128,1)
                #     plot_batch(grad_b[np.newaxis,:,:,:], vis_dir + '/_grad_inv_conv_'+str(b)+'.png')
                #     grad[b] += grad_b


                #grad = grad[:,:,:,np.newaxis]
                #plot_batch(grad, vis_dir + '/_grad_inv_conv_' + str(i) + '.png')
                dtv = grad_tv(img_batch, img_batch0)
                dtv = dtv.reshape(64,128,128,1)
                grad += 300.*dtv
                #grad = grad.reshape(64,128,128,1)
                img_batch -= 1e-5*grad
                restored_images = img_batch
                pred_res = np.abs(restored_images - img_batch0)
                if r%10==0:
                    plot_batch(restored_images, vis_dir + '/_images_restored_' + str(r) + '.png')
                    plot_batch(np.abs(restored_images - img_batch0), vis_dir + '/_images_dif_' + str(i) + '.png')

            return restored_images, pred_res

    def gaussian_approximator_recons_var(self, weight, riter, prior_residuals,
                              dec_mu, constraint='L1'):
        t_vars = tf.trainable_variables()
        update_vars = [var for var in t_vars if 'img_ano' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # self.gfunc = tf.reduce_mean(self.autoencoder_l2/(self.res**2+1e-10) + weight*tf.image.total_variation(
        #     tf.abs(self.input_image-self.img_ano)))
        # self.gfunc = tf.reduce_mean(self.autoencoder_l2 * tf.exp(self.res) + weight * tf.image.total_variation(
        #    tf.abs(self.input_image - self.img_ano)))
        #
        # self.gfunc = tf.reduce_mean(self.autoencoder_l2_weight) + weight * tf.reduce_mean(losses.l1loss(
        #     self.input_image, self.img_ano)) + weight*self.loss_residual

        self.gfunc = tf.reduce_mean(self.autoencoder_l2_weight)
        if "L1" in constraint:
             self.gfunc+=weight * tf.reduce_mean(losses.l1loss(
                self.input_image, self.img_ano))
             if "combined" in constraint:
                 self.gfunc+= self.loss_residual

        # TV constraint
        elif "TV" in constraint:
            self.gfunc += weight * tf.reduce_mean(tf.image.total_variation(self.img_ano - self.input_image))
            if "combined" in constraint:
                self.gfunc+= self.loss_residual
        # self.gfunc = tf.reduce_mean(self.autoencoder_l2_weighted) + weight*tf.reduce_mean(losses.l1loss(
        #    self.input_image, self.img_ano)) #+ 20.*self.loss_residual
        # self.gfunc = self.loss_vae + weight*tf.reduce_mean(losses.l1loss(self.input_image, self.img_ano))

        # self.gfunc = tf.reduce_mean(self.autoencoder_l2 *tf.exp(self.res) + weight * losses.l1loss(
        #    self.input_image, self.img_ano)) + 15. * self.loss_residual
        # 1. / np.sqrt(np.exp(self.out_std_test)),

        with tf.control_dependencies(update_ops):
            adam = tf.train.AdamOptimizer(0.005)  # tv: 0.005  l1: 0.005  probl1:0.01
            grad_vars = adam.compute_gradients(self.gfunc, var_list=update_vars)
            self.grad = [tf.clip_by_value(grad * self.img_mask, -5., 5.) for grad, var in grad_vars]
            #restore_op = adam.apply_gradients([(grad_vars[0][0] * self.img_mask, grad_vars[0][1])])
            restore_op = adam.apply_gradients([(grad_vars[0][0] * self.img_mask,  grad_vars[0][1])])

        initialize_uninitialized(self.sess)
        for i in range(riter):
            decoded_concat = []
            decoded_mu = np.zeros(dec_mu.shape)
            for s in range(10):
                recons_samples = self.sess.run(self.decoder_output)
                decoded_concat.append(recons_samples)
                # error_samples = np.abs(recons_samples - img_batch)
                # decoded.append(recons_samples)
                decoded_mu += recons_samples
                #plot_batch(recons_samples,
                #           vis_dir + '/reconstruct_' + str(s) + '.png')

            decoded_mu = decoded_mu / 10
            decoded_concat = np.transpose(decoded_concat, [1, 2, 3, 4, 0])
            decoded_concat = decoded_concat.reshape(64, 128, 128, 10)

            decoded_std = np.std(decoded_concat, axis=3, keepdims=True)
            decoded_std[self.mask_image==0]=0

            loss_vae, pred_res, _ = self.sess.run([self.loss_vae, self.pred_res, restore_op],
                                                  {self.res: decoded_std,
                                                   self.res_mu: decoded_mu})

            loss_residual, pred_res = self.sess.run([self.loss_residual, self.pred_res],
                                                    {self.res: decoded_std,
                                                     self.res_mu: decoded_mu})

            # loss_vae, pred_res, _ = self.sess.run([self.loss_vae, self.pred_res, restore_op],
            #                                      {self.res: prior_residuals})

            # loss_residual, pred_res = self.sess.run([self.loss_residual, self.pred_res],
            #                                               {self.res: prior_residuals})

            if i % 100 == 0:
                print(loss_vae, loss_residual)
                w_clip = self.sess.run(self.w, {self.res: decoded_std,
                                                self.res_mu: dec_mu})
                print(np.percentile(w_clip.ravel(), 1),
                      np.percentile(w_clip.ravel(), 20),
                      np.percentile(w_clip.ravel(), 76),
                      np.percentile(w_clip.ravel(), 95),
                      np.percentile(w_clip.ravel(), 99),
                      np.percentile(w_clip.ravel(), 99.9),
                      np.max(w_clip.ravel()))
                restored_images = self.sess.run(self.img_ano)
                plot_batch(pred_res, vis_dir + '/_pred_restored_' + str(i) + '.png')
                plot_batch(prior_residuals, vis_dir + '/_true_pred_restored_' + str(i) + '.png')
                plot_batch(decoded_std, vis_dir + '/_std_' + str(i) + '.png')

                plot_batch(restored_images, vis_dir + '/_images_restored_' + str(i) + '.png')
                # plot_batch(restored_mask, vis_dir + '/_masks_restored_' + str(i) + '.png')
        return restored_images, pred_res

def norm_prob_log(m, v, z, k):
    logp1 = np.log(2 * np.pi) ** (-k / 2) * (np.linalg.det(v) ** (-1 / 2))
    logp2 = np.dot((z - m).T, np.linalg.inv(v))
    logp2 = -0.5 * np.dot(logp2, z - m)
    return logp1+logp2

def prob_zi_x(mean, cov, z):
    logp=0
    k = mean.shape[-1]
    for m, v, z_i in zip(mean, cov, z):
        logp_i = norm_prob_log(m, v, z_i, k)
        logp += logp_i
    return np.exp(logp)

def prob_z_x(mean_tensor, cov_tensor, z_tensor):

    mean_tensor = mean_tensor.reshape(64,256,4)
    z_tensor = z_tensor.reshape(64,256,4)
    prob_z_x_array = []
    for mt, ct, zt in zip(mean_tensor, cov_tensor, z_tensor):
        pzi_x = prob_zi_x(mt, ct, zt)
        prob_z_x_array.append(pzi_x)
    return prob_z_x_array

def importance_weight(mean, cov, z_array):
    pz_x = np.array(prob_z_x(mean, cov, z_array))

    z_array = z_array.reshape(64, 256,4)

    m = np.zeros_like(z_array)
    v = np.diag([1,1,1,1])
    p_z = []
    for mt, zt in zip(m, z_array):
        k = mean.shape[-1]
        logp=0
        for m, z_i in zip(mt, zt):
            logp_i = norm_prob_log(m, v, z_i, k)
            logp += logp_i
        pzi = np.exp(logp)
        p_z.append(pzi)

    p_z = np.array(p_z)

    w = p_z/pz_x

    return w


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=0)
    parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--test_files', type=str)
    parser.add_argument('--restore_method', type=str, default='neural')
    parser.add_argument('--restore_constraint', type=str, default='L1')
    parser.add_argument('--fprate', type=float)
    parser.add_argument('--weight', type=float)
    #parser.add_argument('--threshold_val', type=float)
    parser.add_argument('--preset_threshold', nargs='+', type=float, default=None)
    parser.add_argument("--config", required = True, help = "path to config")
    parser.add_argument("--checkpoint", default = "logs", help = "path to checkpoint to restore")
    parser.set_defaults(retrain = False)

    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)

    save_auc=False
    save_dsc=False

    model_name = opt.model_name
    step = opt.load_step
    test_filepath = opt.test_files
    fprate = opt.fprate
    weight = opt.weight
    restore_method = opt.restore_method
    constraint = opt.restore_constraint
    #threshold_val = opt.threshold_val
    threshold_val = opt.preset_threshold

    epochs = config['lr_decay_end']
    batch_size = config["batch_size"]
    img_size = config["spatial_size"]
    img_shape = 2 * [config["spatial_size"]] + [1]
    data_shape = [batch_size] + img_shape
    init_shape = [config["init_batches"] * batch_size] + img_shape
    box_factor = config["box_factor"]
    data_index = config["data_index"]
    LR = config["lr"]

    log_dir = "logs"

    thresh_error = []
    # thresh_error_corr = []
    # thresh_MAD_corr = []
    auc_pred = []
    auc_true = []

    #dsc_array = np.zeros(4)
    dsc_arr = []
    dsc_arr2 = []
    dsc_arr3 = []
    dsc_arr4 = []
    dsc_dev_array = np.zeros(3)
    sub_num = 1.

    total_p = 0
    total_n = 0

    n_latent_samples = 10

    if ".p" in test_filepath:
        test_file = open(test_filepath, "rb")
        import pickle
        test_subs = pickle.load(test_file)[0]
    else:
        test_file = open(test_filepath, "r")
        test_subs = test_file.readlines()

    dsc_array = np.zeros(len(threshold_val))

    tf.reset_default_graph()

    vae_network = VariationalAutoencoder
    model = VAEModel(vae_network, config, model_name, log_dir)
    model.load(model_name, step)

    l = len(test_subs)

    for test_sub in test_subs:
        if ".p" in test_filepath:
            test_sub = "brats/"+test_sub.split("mni/")[-1]
        else:
            test_sub = test_sub[:-1]
        print(test_sub)
        print(sub_num)

        # test outer loop
        img_filename = test_sub.split("\n")[0]
        #img_filename = img_filename.replace('normalized','renormalized')
        # img = nib.load(img_filename).get_data()
        # seg = img_filename.split('normalized')[0] + "seg_cropped.nii.gz"
        # seg = nib.load(seg).get_data()
        # seg[seg != 0] = 1
        #
        # mask = 'mask'.join(img_filename.split('normalized'))
        # mask = nib.load(mask).get_data()
        #
        # subject_name = img_filename.split('_normalized')[0]
        #image_original_size = img.shape[1]

        # img_filename = "_unbiased_eq_normed".join(test_sub.split("_normalized_cropped_mask"))
        # img = nib.load(img_filename).get_data()
        # mask = 'mask'.join(img_filename.split('normed'))
        # mask = nib.load(mask).get_data()
        #
        # seg = 'seg'.join(img_filename.split('normed'))
        # seg = nib.load(seg).get_data()
        # seg[seg>0]=1
        #
        # subject_name = img_filename.split('-mni')[-1]
        #
        # img = np.rot90(img, 3, (2, 1))
        # seg = np.rot90(seg, 3, (2, 1))
        # mask = np.rot90(mask, 3, (2, 1))

        # img = img[:, 22:180, 17:215]
        # mask = mask[:, 22:180, 17:215]
        # seg = seg[:, 22:180, 17:215]

        # img = img[:, 25:215, 30:220]
        # mask = mask[:, 25:215, 30:220]
        # seg = seg[:, 25:215, 30:220]

        img = nib.load(img_filename).get_data()
        seg = img_filename.split('normalized')[0] + "seg_cropped.nii.gz"
        seg = nib.load(seg).get_data()
        seg[seg != 0] = 1

        mask = 'mask'.join(img_filename.split('normalized'))
        mask = nib.load(mask).get_data()

        subject_name = img_filename.split('_normalized')[0]
        image_original_size = img.shape[1]

        image_original_size = 200

        #if restore_method=="neural":
        if ".p" not in test_filepath:
            try:
                os.makedirs("tests/"+str(model_name) + '_' +str(restore_method) +'_' +str(constraint) +'_' +str(weight)+'_rerun/'
                            + str(subject_name))
            except OSError:
                pass
            vis_dir = "tests/"+str(model_name) + '_' +str(restore_method) +'_' +str(constraint) +'_' +str(weight)+'_rerun/' \
                      + str(subject_name)

        elif ".p" in test_filepath:
            try:
                os.makedirs(
                    "tests/" + str(model_name) + '_' + str(restore_method) + '_' + str(constraint) + '_' + str(weight) + 'p/'
                    + str(subject_name))
            except OSError:
                pass
            vis_dir = "tests/" + str(model_name) + '_' + str(restore_method) + '_' + str(constraint) + '_' + str(weight) + 'p/' \
                      + str(subject_name)

        len0 = len(img)
        img_minval = img.min()

        if len0%batch_size:
            fill_len = (int(len0/batch_size)+1)*batch_size-len0

            fill0 = np.zeros((fill_len, img_size, img_size))
            fill_img = np.zeros((fill_len, image_original_size, image_original_size))+img[:fill_len]
            fill_mask = np.zeros((fill_len, image_original_size, image_original_size)) + mask[:fill_len]

            img_filled = np.append(img, fill_img, axis=0)
            mask_filled = np.append(mask,
                                    np.zeros((fill_len, image_original_size, image_original_size)),
                                    axis=0)
        else:
            img_filled = img
            mask_filled = mask

        img_filled = resize(img_filled, img_size / image_original_size, "bilinear")
        mask_filled = resize(mask_filled, img_size / image_original_size, "nearest")
        img_len = int(len(img_filled)/batch_size)

        error_sub = []
        error_corr_sub = []
        MAD_sub = []

        f_errors_total = []
        MAD_total = []

        for i in range(img_len):
            samples_avg = 0
            samples_sq_avg = 0

            img_batch = img_filled[i * batch_size:(i + 1) * batch_size]
            mask_batch = mask_filled[i * batch_size:(i + 1) * batch_size]
            img_batch = img_batch[:, :, :, np.newaxis]
            mask_batch = mask_batch[:, :, :, np.newaxis]

            model.validate(img_batch,1)
            recons_samples = model.out_mu_test
            prior_residuals = model.residual_output_test
            plot_batch(prior_residuals[-1:,:,:,:], vis_dir + '/pred_error_' + str(i) + '.png')
            error_br = np.abs(img_batch - recons_samples)
            # grad = error_br/prior_residuals
            # plot_batch(grad,vis_dir + '/grad_error_' + str(i) + '.png' )
            # decoded_mu = np.zeros(img_batch.shape)
            # decoded_concat = []
            # w_im_array = []
            # for s in range(n_latent_samples):
            #     model.validate(img_batch)
            #     recons_samples = model.out_mu_test
            #     z_sample = model.guessed_z
            #     w_im = importance_weight(z_sample)
            #     w_im_array.append(w_im)
            #     decoded_concat.append(recons_samples)
            #     #error_samples = np.abs(recons_samples - img_batch)
            #     #decoded.append(recons_samples)
            #     decoded_mu += w_im*recons_samples
            #     #plot_batch(recons_samples,
            #     #           vis_dir + '/reconstruct_' + str(s) + '.png')
            #     #decoded += (recons_samples - img_batch)**2
            #
            # #stddev_samples = model.out_std_test
            #
            # residuals = model.residual_output_test
            # #decoded.append(residuals)
            #
            # #decoded = np.asarray(decoded).reshape(n_latent_samples, batch_size, img_size, img_size)
            # #decoded = np.asarray(decoded).reshape(n_latent_samples, batch_size, img_size, img_size)
            # #decoded = np.transpose(decoded, (1, 2, 3, 0))
            # decoded_mu = decoded_mu/n_latent_samples
            #
            # decoded_concat = np.transpose(decoded_concat,[1,2,3,4,0])
            # decoded_concat = decoded_concat.reshape(64,128,128,n_latent_samples)
            # decoded_std = np.std(decoded_concat,axis=3,keepdims=True)

            #plot_batch(decoded_std,
            #           vis_dir + '/std.png')

            print(np.sum(mask_batch))
            if np.sum(mask_batch)>30000:
                if restore_method=="neural":
                    #prior_residuals = np.array([np.diag(i) for i in prior_residuals]
                    start = time.time()
                    # for i in range(800):
                    decoded_mu = np.zeros(img_batch.shape)
                    decoded_concat = []
                    w_im_array = []
                    # for s in range(n_latent_samples):
                    #     model.validate(img_batch)
                    #     recons_samples = model.out_mu_test
                    #     z_sample, z_mean, z_cov = model.sess.run([model.guessed_z,
                    #                                model.z_mean,
                    #                                model.z_A_T],
                    #                               {model.image_matrix: img_batch})
                    #     # z_sample = z_sample.reshape(64,-1)
                    #     w_im = importance_weight(z_mean, z_cov, z_sample)
                    #     w_im = w_im/np.sum(w_im)
                    #     w_im_array.append(w_im)
                    #     #decoded_concat.append(recons_samples)
                    #     decoded_mu += np.array([w_im[i] * recons_samples[i] for i in range(64)])

                    # for s in range(n_latent_samples):
                    #     # model.validate(img_batch)
                    #     recons_samples = model.sess.run(model.decoder_output,
                    #                                     {model.image_matrix: img_batch})
                    #     z_sample, z_mean, z_cov = \
                    #         model.sess.run([model.guessed_z,
                    #                         model.z_mean,
                    #                         model.z_A_T],
                    #                        {model.image_matrix:img_batch})
                    #     w_im = importance_weight(z_mean, z_cov, z_sample)
                    #     w_im_array.append(w_im)
                    #     decoded_concat.append(recons_samples)
                    # w_im_array = np.array(w_im_array)
                    # decoded_concat = np.array(decoded_concat)
                    # w_im_sum = np.sum(w_im_array, axis=0)
                    # w_im_array = np.array([w_im_array[:, i] / w_im_sum[i] for i in range(64)])
                    # w_im_array = w_im_array.T
                    # #decoded_mu = decoded_mu/n_latent_samples
                    #
                    # for i in range(n_latent_samples):
                    #     decoded_mu += np.array([w_im_array[i, j] * decoded_concat[i, j]
                    #                             for j in range(64)])

                    decoded_mu = model.sess.run(model.decoder_output,
                                                         {model.image_matrix: img_batch})

                    #if pred:
                    residuals = model.residual_output_test

                    # elif estimate:
                    #     decoded_concat = np.transpose(decoded_concat, [1, 2, 3, 4, 0])
                    #     decoded_concat = decoded_concat.reshape(64, 128, 128, n_latent_samples)
                    #     decoded_std = np.std(decoded_concat, axis=3, keepdims=True)
                    #     cov = np.diag(decoded_std)
                    #
                    #     mu_flatten = (decoded_mu-recons_samples).T
                    #     cov_mu = np.dot(mu_flatten, mu_flatten.T)
                    #
                    #     cov_all = w_im(cov + cov_mu)/n_latent_samples

                    image_restoration = Restoration(img_batch, mask_batch, vae_network, model_name, config)
                    image_restoration.reconstructor_vae()
                    image_restoration.load(log_dir, model_name, step)
                    restored_images, pred_res_restored = \
                     image_restoration.gaussian_approximator(weight, 1500, prior_residuals, decoded_mu,
                                                             constraint=constraint)
                    #restored_images, pred_res_restored = \
                    #image_restoration.gaussian_approximator_estimate_cov(img_batch, weight, 200,
                    #                                        constraint=constraint)
                    end = time.time()
                    print("runtime {}".format(end-start))

                elif restore_method=="cov_estimate":
                    print(time.time())
                    # for i in range(800):
                    # decoded_mu = np.zeros(img_batch.shape)
                    # decoded_concat = []
                    # w_im_array = []
                    # for s in range(n_latent_samples):
                    #     model.validate(img_batch)
                    #     recons_samples = model.out_mu_test
                    #     z_sample, z_mean, z_cov = model.sess.run([model.guessed_z,
                    #                                model.z_mean,
                    #                                model.z_A_T],
                    #                               {model.image_matrix: img_batch})
                    #     # z_sample = z_sample.reshape(64,-1)
                    #     w_im = importance_weight(z_mean, z_cov, z_sample)
                    #     w_im = w_im/np.sum(w_im)
                    #     w_im_array.append(w_im)
                    #     #decoded_concat.append(recons_samples)
                    #     decoded_mu += np.array([w_im[i] * recons_samples[i] for i in range(64)])

                    # for s in range(n_latent_samples):
                    #     # model.validate(img_batch)
                    #     recons_samples = model.sess.run(model.decoder_output,
                    #                                     {model.image_matrix: img_batch})
                    #     z_sample, z_mean, z_cov = \
                    #         model.sess.run([model.guessed_z,
                    #                         model.z_mean,
                    #                         model.z_A_T],
                    #                        {model.image_matrix:img_batch})
                    #     w_im = importance_weight(z_mean, z_cov, z_sample)
                    #     w_im_array.append(w_im)
                    #     decoded_concat.append(recons_samples)
                    # w_im_array = np.array(w_im_array)
                    # decoded_concat = np.array(decoded_concat)
                    # w_im_sum = np.sum(w_im_array, axis=0)
                    # w_im_array = np.array([w_im_array[:, i] / w_im_sum[i] for i in range(64)])
                    # w_im_array = w_im_array.T
                    # #decoded_mu = decoded_mu/n_latent_samples
                    #
                    # for i in range(n_latent_samples):
                    #     decoded_mu += np.array([w_im_array[i, j] * decoded_concat[i, j]
                    #                             for j in range(64)])

                    #if pred:
                    residuals = model.residual_output_test
                        #cov_residuals = np.diag(residuals)

                    # elif estimate:
                    #     decoded_concat = np.transpose(decoded_concat, [1, 2, 3, 4, 0])
                    #     decoded_concat = decoded_concat.reshape(64, 128, 128, n_latent_samples)
                    #     decoded_std = np.std(decoded_concat, axis=3, keepdims=True)
                    #     cov = np.diag(decoded_std)
                    #
                    #     mu_flatten = (decoded_mu-recons_samples).T
                    #     cov_mu = np.dot(mu_flatten, mu_flatten.T)
                    #
                    #     cov_all = w_im(cov + cov_mu)/n_latent_samples

                    image_restoration = Restoration(img_batch, mask_batch, vae_network, model_name, config)
                    image_restoration.reconstructor()
                    image_restoration.load(log_dir, model_name, step)
                    restored_images, pred_res_restored = \
                     image_restoration.gaussian_approximator(weight, 800, prior_residuals, decoded_mu,
                                                             constraint=constraint)
                    #restored_images, pred_res_restored = \
                    #image_restoration.gaussian_approximator_estimate_cov(img_batch, weight, 200,
                    #                                       constraint=constraint)
                    end = time.time()
                    print("runtime {}".format(end-start))

                elif restore_method=="estimate":
                    start = time.time()
                    for i in range(800):
                        decoded_mu = np.zeros(img_batch.shape)
                        decoded_concat = []
                        w_im_array = []
                        for s in range(n_latent_samples):
                            model.validate(img_batch)
                            recons_samples = model.out_mu_test
                            z_sample = model.guessed_z
                            w_im = importance_weight(z_sample)
                            w_im_array.append(w_im)
                            decoded_concat.append(recons_samples)
                            decoded_mu += w_im * recons_samples

                        #decoded_mu = decoded_mu / n_latent_samples

                        decoded_concat = np.transpose(decoded_concat, [1, 2, 3, 4, 0])
                        decoded_concat = decoded_concat.reshape(64, 128, 128, n_latent_samples)
                        decoded_std = np.std(decoded_concat, axis=3, keepdims=True)
                        cov = np.diag(decoded_std)

                        mu_flatten = (decoded_mu-recons_samples).T
                        cov_mu = np.dot(mu_flatten, mu_flatten.T)

                        cov_all = w_im(cov + cov_mu)/n_latent_samples

                        image_restoration = Restoration(img_batch, mask_batch, vae_network, model_name, config)
                        image_restoration.reconstructor()
                        image_restoration.load(log_dir, model_name, step)
                        restored_images, pred_res_restored = \
                           image_restoration.gaussian_approximator_recons_var(weight, 800, prior_residuals, decoded_mu,
                                                                   constraint=constraint)
                    end = time.time()
                    print("runtime {}".format(end-start))
            else:
                restored_images = img_batch
                pred_res_restored = np.zeros(img_batch.shape)
            # 30000 for tv norm
            # 8000

            error_restored = np.abs(restored_images - img_batch)
            #error_restored = error_restored.max()-error_restored

            error_sub.extend(error_restored)

            # plot_batch(recons_samples,
            #            vis_dir + '/reconstruct_' + str(i) + '.png')
            # plot_batch(prior_residuals,
            #            vis_dir + '/pred_error_' + str(i) + '.png')
            # plot_batch(pred_res_restored,
            #            vis_dir + '/pred_error_restored' + str(i) + '.png')
            # plot_batch(error_br,
            #            vis_dir + '/error_no-restore_' + str(i) + '.png')
            # plot_batch(error_restored, vis_dir + '/error_restored_' + str(i) + '.png')
            # plot_batch(restored_images, vis_dir + '/images_restored_' + str(i) + '.png')
            # plot_batch(img_batch, vis_dir + '/images_' + str(i) + '.png')
            plot_batch(error_restored, vis_dir + '/error_restored_' + str(i) + '.png')
            plot_batch(restored_images, vis_dir + '/images_restored_' + str(i) + '.png')
            plot_batch(img_batch, vis_dir + '/images_' + str(i) + '.png')

        #plot_batch(np.asarray(error_sub[:len0]), vis_dir + '/error_restored.png')
        #plot_batch(np.asarray(seg), vis_dir + '/seg.png')
        plot_batch(np.asarray(seg[63:64, :, :, np.newaxis]), vis_dir + '/seg.png')
        error_sub = np.reshape(error_sub[:len0], [-1, img_size, img_size])
        error_sub = resize(error_sub, image_original_size / img_size, method="bilinear")

        error_sub_m = error_sub[mask == 1].ravel()
        seg_m = seg[mask == 1].ravel()

        if not len(thresh_error):
            thresh_error = np.concatenate((np.sort(error_sub_m[::100]), [15]))
            error_tprfpr = np.zeros((2, len(thresh_error)))


        error_tprfpr += compute_tpr_fpr(seg_m, error_sub_m, thresh_error)
        print(error_sub_m.min(), error_sub_m.max(), np.percentile(error_sub_m,90))

        total_p += np.sum(seg_m == 1)
        total_n += np.sum(seg_m == 0)

        tpr_error = error_tprfpr[0] / total_p
        fpr_error = error_tprfpr[1] / total_n

        auc_error = 1. + np.trapz(fpr_error, tpr_error)

        print(test_sub)

        if save_auc:
            np.savetxt("AUCs/tpr_error_"+str(restore_method)+"_"+
                       str(constraint)+".txt", tpr_error)
            np.savetxt("AUCs/fpr_error_" + str(restore_method) + "_" +
                       str(constraint) + ".txt", fpr_error)
        #np.savetxt("fpr_error_l1.txt",error_tprfpr[1])
        print(auc_error, total_p, total_n)

        cutoff_idx = np.argmax(tpr_error - fpr_error)
        cutoff_error = thresh_error[cutoff_idx]
        print("AUC_error:{}, cutoff_value:{}".format(auc_error, cutoff_error))
        print("cutoff_value tpr {}, tpr {}".format(
            tpr_error[cutoff_idx],
            fpr_error[cutoff_idx]))

        # for val in threshold_val:
        #     dsc_error = dsc(error_sub_m > val, seg_m)
        #     dsc_arr.append(dsc_error)
        #
        # dsc_array += np.array(dsc_arr)
        #
        dsc_error = dsc(error_sub_m > threshold_val[0], seg_m)
        print(dsc_error)
        dsc_arr.append(dsc_error)

        dsc_error = dsc(error_sub_m > threshold_val[1], seg_m)
        print(dsc_error)
        dsc_arr2.append(dsc_error)

        dsc_error = dsc(error_sub_m > threshold_val[2], seg_m)
        print(dsc_error)
        dsc_arr3.append(dsc_error)

        dsc_error = dsc(error_sub_m > threshold_val[3], seg_m)
        print(dsc_error)
        dsc_arr4.append(dsc_error)

        #print(np.array(dsc_arr))
        # print(weight)
        #print(dsc_error)
        print([
            np.mean(dsc_arr), np.mean(dsc_arr2), np.mean(dsc_arr3),
            np.mean(dsc_arr4)
        ])
        sub_num += 1.

        if save_dsc:
            np.savetxt("DSCs/" + str(weight) + "_" + str(restore_method) + "_" +
                       str(constraint) + "_dsc_005.txt", dsc_arr)
            np.savetxt("DSCs/" + str(weight) + "_" + str(restore_method) + "_" +
                       str(constraint) + "_dsc_001.txt", dsc_arr2)
            np.savetxt("DSCs/" + str(weight) + "_" + str(restore_method) + "_" +
                       str(constraint) + "_dsc_0005.txt", dsc_arr3)
            np.savetxt("DSCs/" + str(weight) + "_" + str(restore_method) + "_" +
                       str(constraint) + "_dsc_0001.txt", dsc_arr4)
