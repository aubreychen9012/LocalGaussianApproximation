import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SGE_GPU']
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import nibabel as nib
from scipy import interpolate
# tf.enable_eager_execution()
import yaml
import argparse
import h5py
import random

#from networks.res_bilinear_covvae import VariationalAutoencoder
from networks.vae_res_bilinear_conv import VariationalAutoencoder

#from models.covVAE import covVAEModel as VAEModel
from models.vae import VAEModel
from utils import losses
from utils.threshold import *
from utils.batches import get_camcan_batches, tile, plot_batch
from utils.norms import total_variation
from pdb import set_trace as bp
from skimage.measure import compare_ssim as ssim

from preprocess.preprocess import *


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    # print([str(i.name) for i in not_initialized_vars])
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
        # self.model.__init__

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
        # self.res_mu = tf.placeholder('float32', [self.config["batch_size"], self.config["spatial_size"],
        #                                          self.config["spatial_size"], 1],
        #                              name='residuals_mu')

        z_mean, z_A_T, self.pred_res = \
            self.model.encoder(self.img_ano, is_train=True, reuse=False)
        # z_stddev = tf.matmul(z_A, tf.transpose(z_A))
        samples = tf.random_normal(tf.shape(z_mean), 0., 1., dtype=tf.float32)
        guessed_z = z_mean + tf.matmul(samples, z_A_T)  # + z_A_T*samples #
        self.decoder_output, _ = \
            self.model.decoder(guessed_z, name="img", is_train=True, reuse=False)
        self.autoencoder_loss = losses.l2loss(self.decoder_output, self.img_ano)

        # self.residuals = tf.abs(self.img_ano - self.decoder_output)
        self.autoencoder_res_loss = losses.l2loss(self.res, self.pred_res)

        #weighted_l2 = (self.decoder_output - self.img_ano) ** 2 / (self.res ** 2 + 1e-10)
        # clip_value = tf.contrib.distributions.percentile(weighted_l2, q=99.5)

        ##weighted_l2 = tf.clip_by_value(weighted_l2, -5e3, 5e3)
        #self.w = (self.decoder_output - self.img_ano) ** 2 / (self.res ** 2 + 1e-10)
        # self.autoencoder_l2_weight = tf.reduce_sum(weighted_l2 / (self.res),
        #                                           axis=[1, 2, 3])

        #self.autoencoder_l2_weighted = tf.reduce_sum(weighted_l2, axis=[1, 2, 3])
        # self.loss_m = 10.*tf.reduce_sum(tf.abs(self.img_mask), axis=[1,2,3])

        # nd KL
        # self.latent_loss = losses.kl_loss_1d(z_mean, z_A_T)
        self.latent_loss = losses.kl_cov_gaussian(z_mean, z_A_T)
        # self.loss = tf.reduce_mean(self.autoencoder_loss + self.latent_loss
        #                           + self.autoencoder_res_loss + self.loss_m)

        self.loss_vae = tf.reduce_mean(self.autoencoder_loss + self.latent_loss)  # + self.loss_m)
        self.loss_residual = tf.reduce_mean(self.autoencoder_res_loss)

        t_vars = tf.trainable_variables()
        load_vars = [var for var in t_vars if "encoder" in var.name or "decoder" in var.name]
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

        # z_mean_valid, z_std_valid, self.res_test = self.model.encoder(self.image_matrix,
        #                                                               is_train=False, reuse=True)
        # # z_stddev_valid = tf.matmul(z_A_valid, tf.transpose(z_A_valid))
        # samples_valid = tf.random_normal(tf.shape(z_mean_valid), 0., 1., dtype=tf.float32)
        # guessed_z_valid = z_mean_valid + z_std_valid * samples_valid
        # self.decoder_output_test = self.model.decoder(guessed_z_valid,
        #                                               name="img", is_train=False, reuse=True)
        self.autoencoder_loss = losses.l1loss(self.decoder_output, self.img_ano)

        # self.residuals = tf.abs(self.img_ano - self.decoder_output)
        self.autoencoder_res_loss = losses.l1loss(self.res, self.pred_res)
        weighted_l2 = (self.res_mu - self.img_ano) ** 2 / (self.res ** 2 + 1e-10)

        weighted_l2 = tf.clip_by_value(weighted_l2, -5e3, 5e3)
        self.w = (self.res_mu - self.img_ano) ** 2 / (self.res ** 2 + 1e-10)
        self.autoencoder_l2_weight = tf.reduce_sum(weighted_l2,
                                                   axis=[1, 2, 3])
        # nd KL
        self.latent_loss = losses.kl_loss_1d(self.z_mean, self.z_std)

        self.loss_vae = tf.reduce_mean(self.autoencoder_loss + self.latent_loss)  # + self.loss_m)
        self.loss_residual = tf.reduce_mean(self.autoencoder_res_loss)

        t_vars = tf.trainable_variables()
        load_vars = [var for var in t_vars if "encoder" in var.name or "decoder" in var.name]
        self.saver = tf.train.Saver(load_vars)

    def load(self, log_dir, model_name, step):
        model_folder = os.path.join(log_dir, model_name)
        self.saver.restore(self.sess, model_folder + '/' + model_name + ".ckpt-" + str(step))

    def run(self, prior_residuals, update_lr, restore_iteration):
        riter = restore_iteration
        t_vars = tf.trainable_variables()
        update_vars = [var for var in t_vars if 'img_ano' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            adam = tf.train.AdamOptimizer(0.05)
            grad_vars = adam.compute_gradients(self.loss_vae, var_list=update_vars)
            # img_mask = tf.constant(img_mask, dtype=tf.float32)
            # masked_gradients = grad_vars[0][0]*img_mask
            restore_op = adam.apply_gradients([(grad_vars[0][0] * self.img_mask, grad_vars[0][1])])
            grad_vars = adam.compute_gradients(self.loss_residual, var_list=update_vars)
            residual_op = adam.apply_gradients([(grad_vars[0][0] * self.img_mask, grad_vars[0][1])])
            # restore_op = tf.train.AdamOptimizer(0.1).minimize(self.loss,
            #                                                        var_list=update_vars)
        initialize_uninitialized(self.sess)
        for i in range(riter):
            loss_vae, pred_res, _ = self.sess.run([self.loss_vae, self.pred_res, restore_op],
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
                # restored_images[self.mask_image == 0] = -3.5
                # dev = restored_images.min()+3.5
                # restored_images -=dev
                plot_batch(restored_images, vis_dir + '/_images_restored_' + str(i) + '.png')
                # plot_batch(restored_mask, vis_dir + '/_masks_restored_' + str(i) + '.png')

        return restored_images, pred_res


    def tv_approximator(self, weight, riter, prior_residuals, constraint='TV'):
        t_vars = tf.trainable_variables()
        update_vars = [var for var in t_vars if 'img_ano' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # self.gfunc = self.loss_vae + \
        #              weight * tf.reduce_mean(tf.image.total_variation(self.img_ano - self.input_image))
        #
        if constraint == "L1":
            self.gfunc = self.loss_vae \
                         + weight * tf.reduce_mean(losses.l1loss(
                self.input_image, self.img_ano)) #+ self.loss_residual

        # TV constraint
        elif constraint == "TV":
            self.gfunc = self.loss_vae \
                         + weight * tf.reduce_mean(tf.image.total_variation(self.img_ano - self.input_image)) \
                         #+ self.loss_residual


        with tf.control_dependencies(update_ops):
            adam = tf.train.AdamOptimizer(0.005)  # tv: 0.005  l1: 0.005  probl1:0.01
            grad_vars = adam.compute_gradients(self.gfunc, var_list=update_vars)
            self.grad = [tf.clip_by_value(grad * self.img_mask, -100., 100.) for grad, var in grad_vars]
            restore_op = adam.apply_gradients([(tf.clip_by_value(grad_vars[0][0] * self.img_mask, -100., 100.),
                                                grad_vars[0][1])])

        initialize_uninitialized(self.sess)
        for i in range(riter):
            loss_vae, pred_res, _ = self.sess.run([self.loss_vae, self.pred_res, restore_op],
                                                  {self.res: prior_residuals})

            # loss_residual, pred_res = self.sess.run([self.loss_residual, self.pred_res],
            #                                         {self.res: prior_residuals})

            # loss_vae, pred_res, _ = self.sess.run([self.loss_vae, self.pred_res, restore_op],
            #                                      {self.res: prior_residuals})

            # loss_residual, pred_res = self.sess.run([self.loss_residual, self.pred_res],
            #                                               {self.res: prior_residuals})

            if i % 100 == 0:
                print(loss_vae)
                # w_clip = self.sess.run(self.w, {self.res: prior_residuals})
                # print(np.percentile(w_clip.ravel(), 1),
                #       np.percentile(w_clip.ravel(), 20),
                #       np.percentile(w_clip.ravel(), 76),
                #       np.percentile(w_clip.ravel(), 95),
                #       np.percentile(w_clip.ravel(), 99),
                #       np.percentile(w_clip.ravel(), 99.9),
                #       np.max(w_clip.ravel()))
                restored_images = self.sess.run(self.img_ano)
                plot_batch(pred_res, vis_dir + '/_pred_restored_' + str(i) + '.png')
                plot_batch(prior_residuals, vis_dir + '/_true_pred_restored_' + str(i) + '.png')

                plot_batch(restored_images, vis_dir + '/_images_restored_' + str(i) + '.png')
                # plot_batch(restored_mask, vis_dir + '/_masks_restored_' + str(i) + '.png')
        restored_images = self.sess.run(self.img_ano)
        return restored_images, pred_res


def restore_camcan_tv(weight, log_dir, model_name, step, config, img_size, batch_size,
                      n_random_sub=10, fprate=0., renormalized=False,
                      constraint = 'TV'):
    # if renormalized:
    # n = "_renormalized"
    vae_network = VariationalAutoencoder
    model = VAEModel(vae_network, config, model_name, log_dir)
    model.load(model_name, step)
    data = h5py.File('/scratch_net/bmicdl01/Data/camcan_val_set.hdf5')
    # else:
    #    data = h5py.File('/scratch_net/bmicdl01/Data/brats_healthy_train.hdf5')
    indices = random.sample(range(len(data['Scan']))[::batch_size], n_random_sub)

    image_size = img_size
    image_original_size = 200
    batch_size = batch_size
    rate_sum = 0
    restored_err_list = []
    err_arr = []
    for ind in indices:
        print(ind)
        # ind = 30656
        # 30656
        # print(num, ind)
        res = data['Scan'][ind:ind + batch_size]
        res = res.reshape(-1, image_original_size, image_original_size)
        mask = data['Mask'][ind:ind + batch_size]
        mask = mask.reshape(-1, image_original_size, image_original_size)

        dim_res = res.shape
        image_original_size = res.shape[1]
        res_minval = res.min()

        if dim_res[0] % batch_size:
            dim_res_expand = batch_size - (dim_res[0] % batch_size)
            res_expand = np.zeros((dim_res_expand, dim_res[1], dim_res[2])) + res_minval
            res_exp = np.append(res, res_expand, axis=0)
            mask_exp = np.append(mask, np.zeros((dim_res_expand, dim_res[1], dim_res[2])), axis=0)
        else:
            res_exp = res
            mask_exp = mask

        res_exp = resize(res_exp, image_size / image_original_size,"bilinear")
        mask_exp = resize(mask_exp, image_size / image_original_size, "nearest")

        for batch in tl.iterate.minibatches(inputs=res_exp, targets=mask_exp,
                                            batch_size=batch_size, shuffle=False):
            b_images, b_masks = batch
            b_images = b_images[:, :, :, np.newaxis]
            b_masks = b_masks[:, :, :, np.newaxis]

            # decoded_mu = np.zeros(b_images.shape)
            # for s in range(25):
            #     model.validate(b_images)
            #     recons_samples = model.out_mu_test
            #     # error_samples = np.abs(recons_samples - img_batch)
            #     # decoded.append(recons_samples)
            #     decoded_mu += recons_samples
            #     # decoded += (recons_samples - img_batch)**2
            #
            # # stddev_samples = model.out_std_test
            #
            # # residuals = model.residual_output_test
            # # decoded.append(residuals)
            #
            # # decoded = np.asarray(decoded).reshape(n_latent_samples, batch_size, img_size, img_size)
            # # decoded = np.asarray(decoded).reshape(n_latent_samples, batch_size, img_size, img_size)
            # # decoded = np.transpose(decoded, (1, 2, 3, 0))
            # decoded_mu = decoded_mu / 25.

            model.validate(b_images, 1.)
            recons_samples = model.out_mu_test
            prior_residuals = model.residual_output_test

            image_restoration = Restoration(b_images, b_masks,
                                            vae_network, model_name, config)
            if "01data" not in model_name:
                image_restoration.reconstructor_vae()
            elif "01data" in model_name:
                image_restoration.reconstructor_z4094()
            image_restoration.load(log_dir, model_name, step)
            restored_images, pred_res_restored = \
                image_restoration.tv_approximator(weight, 800, prior_residuals,
                                                  constraint=constraint)

            # _rate = np.abs(restored_images-b_images)/np.abs(b_images)
            # restored = np.sum(np.abs(restored_images), axis=(1,2,3))
            # b_ = np.sum(np.abs(b_images),axis=(1,2,3))
            # x = np.mean(restored/b_)

            for i in range(len(b_images)):
                ssim_val = ssim(b_images[i], restored_images[i],
                                multichannel=True)
                restored_err_list.append(ssim_val)
            print(np.mean(restored_err_list))

            err = np.abs(b_images - restored_images)[b_masks == 1]
            err_arr.extend(err)
            # _rate = np.abs(1-x)
            # _rate_m = _rate
            # rate = np.mean(_rate_m)
            # print(rate)
            # restored_err_list.extend(np.abs(restored_images-b_images)[b_masks==1])
    threshold0 = determine_threshold(err_arr, 0.05)
    threshold1 = determine_threshold(err_arr, 0.01)
    threshold2 = determine_threshold(err_arr, 0.005)
    threshold3 = determine_threshold(err_arr, 0.1)
    return [np.mean(restored_err_list), threshold0, threshold1,
            threshold2, threshold3, fprate]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=0)
    parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--weight', type=float, default=0.)
    parser.add_argument('--constraint', type=str, default='TV')
    # parser.add_argument('--test_files', type=str)
    parser.add_argument('--fprate', type=float)
    # parser.add_argument('--preset_threshold', nargs='+', type=float, default=None)
    parser.add_argument("--config", required=True, help="path to config")
    # parser.add_argument("--checkpoint", default = "logs", help = "path to checkpoint to restore")
    parser.set_defaults(retrain=False)

    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)

    model_name = opt.model_name
    step = opt.load_step
    fprate = opt.fprate
    constraint = opt.constraint

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

    try:
        os.makedirs("tests/" + str(model_name) + '_samples_camcan/')
    except OSError:
        pass
    vis_dir = "tests/" + str(model_name) + '_samples_camcan/'

    # run threshold.py to get threshold values
    # vae_network = VariationalAutoencoder
    # model = VAEModel(vae_network, config, model_name, log_dir)
    # model.load(model_name, step)
    weight = opt.weight
    #for weight in np.arange(4, 10, 1):
        # change_rate = restore_camcan(weight, log_dir, model_name, step, config, img_size = img_size,
        #                             batch_size=batch_size, fprate=fprate)
    change_rate = restore_camcan_tv(weight, log_dir, model_name, step, config, img_size=img_size,
                                        batch_size=batch_size, fprate=fprate,
                                    constraint = constraint)
    print(change_rate, weight)
