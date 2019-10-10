import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SGE_GPU']
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import tensorflow as tf
import numpy as np
import nibabel as nib
from scipy import interpolate
#tf.enable_eager_execution()
import yaml
import argparse
from networks.vae_res_bilinear_conv import VariationalAutoencoder
#from networks.res_bilinear_covvae import VariationalAutoencoder
#from networks.vae_res_bilinar_in import VariationalAutoencoder
#from networks.vae_probX import VariationalAutoencoder
#from networks.vae_prob_predstddev import VariationalAutoencoder

#from models.covVAE import covVAEModel as VAEModel
from models.vae import VAEModel
#from models.vae_prob_model import VAEModel
#from models.vae_prob_stddev import VAEModel
from utils import losses
from utils.batches import get_camcan_batches,tile, plot_batch
from utils import threshold
from pdb import set_trace as bp

from preprocess.preprocess import *
from metrics.auc_score import compute_tpr_fpr
from metrics.dice import dsc, dsc_compute


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v,f) in zip(global_vars, is_not_initialized) if not f]
    #print([str(i.name) for i in not_initialized_vars])
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
        #self.model.__init__

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

        z_mean, z_A_T, self.pred_res = \
                self.model.encoder(self.img_ano, is_train=True, reuse=False)
        # z_stddev = tf.matmul(z_A, tf.transpose(z_A))
        samples = tf.random_normal(tf.shape(z_mean), 0., 1., dtype=tf.float32)
        guessed_z = z_mean + tf.matmul(samples, z_A_T) #z_mean + z_A_T*samples #
        self.decoder_output,_ = \
            self.model.decoder(guessed_z, name="img", is_train=True, reuse=False)
        self.autoencoder_loss = losses.l1loss(self.decoder_output, self.img_ano)

        #self.residuals = tf.abs(self.img_ano - self.decoder_output)
        self.autoencoder_res_loss = losses.l1loss(self.res, self.pred_res)
        weighted_l2 = (self.res_mu-self.img_ano)**2 #/(self.res**2+1e-10)
        clip_value = tf.contrib.distributions.percentile(weighted_l2, q=95)

        #weighted_l2 = tf.clip_by_value(weighted_l2, -5e3, 5e3)
        self.w = (self.res_mu - self.img_ano) ** 2 #/ (self.res**2+1e-10)
        self.autoencoder_l2_weight = tf.reduce_sum(weighted_l2,
                                                   axis=[1,2,3])
        #self.loss_m = 10.*tf.reduce_sum(tf.abs(self.img_mask), axis=[1,2,3])

        # nd KL
        #self.latent_loss = losses.kl_loss_1d(z_mean, z_A_T)
        self.latent_loss = losses.kl_cov_gaussian(z_mean, z_A_T)
        #self.loss = tf.reduce_mean(self.autoencoder_loss + self.latent_loss
        #                           + self.autoencoder_res_loss + self.loss_m)

        self.loss_vae = tf.reduce_mean(self.autoencoder_loss + self.latent_loss) # + self.loss_m)
        self.loss_residual = tf.reduce_mean(self.autoencoder_res_loss)

        t_vars = tf.trainable_variables()
        load_vars = [var for var in t_vars if "encoder" in var.name or "decoder" in var.name]
        self.saver = tf.train.Saver(load_vars)

    def reconstructor_vae(self):
        init_values = tf.constant_initializer(self.input_image)
        self.init_values = tf.constant(self.input_image)
        self.init_values = tf.cast(self.init_values, tf.float32)
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
        self.loss = self.autoencoder_l2_weight + self.latent_loss
        #self.latent_loss = losses.kl_cov_gaussian(self.z_mean, self.z_std)
        #self.loss = tf.reduce_mean(self.autoencoder_loss + self.latent_loss
        #                           + self.autoencoder_res_loss + self.loss_m)

        self.loss_vae = tf.reduce_mean(self.autoencoder_loss + self.latent_loss) # + self.loss_m)
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
            grad_vars = adam.compute_gradients(self.loss_vae, var_list = update_vars)
            #img_mask = tf.constant(img_mask, dtype=tf.float32)
            #masked_gradients = grad_vars[0][0]*img_mask
            restore_op = adam.apply_gradients([(grad_vars[0][0]*self.img_mask, grad_vars[0][1])])
            grad_vars = adam.compute_gradients(self.loss_residual, var_list=update_vars)
            residual_op = adam.apply_gradients([(grad_vars[0][0]*self.img_mask, grad_vars[0][1])])
            #restore_op = tf.train.AdamOptimizer(0.1).minimize(self.loss,
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
                #restored_images[self.mask_image == 0] = -3.5
                #dev = restored_images.min()+3.5
                #restored_images -=dev
                plot_batch(restored_images, vis_dir + '/_images_restored_' + str(i) + '.png')
                #plot_batch(restored_mask, vis_dir + '/_masks_restored_' + str(i) + '.png')

        return restored_images, pred_res

    def run_map(self, weight, riter, prior_residuals, dec_mu):
        t_vars = tf.trainable_variables()
        update_vars = [var for var in t_vars if 'img_ano' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.gfunc = tf.reduce_mean(self.loss) + \
                     weight*tf.image.total_variation(self.img_ano - self.init_values)

        with tf.control_dependencies(update_ops):
            adam = tf.train.AdamOptimizer(0.005) # tv: 0.005  l1: 0.005  probl1:0.01
            grad_vars = adam.compute_gradients(self.gfunc, var_list=update_vars)
            self.grad = [tf.clip_by_value(grad * self.img_mask, -500.,500.) for grad, var in grad_vars]
            #self.grad = tf.gradients([self.gfunc], [self.img_ano])[0]
            restore_op = adam.apply_gradients([(tf.clip_by_value(grad_vars[0][0] * self.img_mask, -100.,100.),
                                               grad_vars[0][1])])

        initialize_uninitialized(self.sess)
        for i in range(riter):
            loss_vae, pred_res, _ = self.sess.run([self.loss_vae, self.pred_res, restore_op],
                                                 {self.res: prior_residuals,
                                                  self.res_mu: dec_mu})

            loss_residual, pred_res = self.sess.run([self.loss_residual, self.pred_res],
                                                          {self.res: prior_residuals,
                                                           self.res_mu: dec_mu})

            if i % 100 == 0:
                print(loss_vae, loss_residual)
                restored_images = self.sess.run(self.img_ano)
                grads = self.sess.run(self.grad, {self.res: prior_residuals, self.res_mu: dec_mu})
                grads = grads[0]
                grads[self.img_mask==0]=0
                w_clip = self.sess.run(self.w, {self.res: prior_residuals,
                                                self.res_mu: dec_mu})
                # print(np.percentile(w_clip.ravel(), 1),
                #       np.percentile(w_clip.ravel(), 20),
                #       np.percentile(w_clip.ravel(), 76),
                #       np.percentile(w_clip.ravel(), 95),
                #       np.percentile(w_clip.ravel(), 99),
                #       np.percentile(w_clip.ravel(), 99.9),
                #       np.max(w_clip.ravel()))

                plot_batch(restored_images, vis_dir + '/_images_restored_' + str(i) + '.png')
                plot_batch(grads, vis_dir + '/_grads_restored_' + str(i) + '.png')
        return restored_images, pred_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=0)
    parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--test_files', type=str)
    parser.add_argument('--fprate', type=float)
    parser.add_argument('--weight', type=float)
    parser.add_argument('--preset_threshold', nargs='+', type=float, default=None)
    parser.add_argument("--config", required = True, help = "path to config")
    parser.add_argument("--checkpoint", default = "logs", help = "path to checkpoint to restore")
    parser.set_defaults(retrain = False)

    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)

    model_name = opt.model_name
    step = opt.load_step
    test_filepath = opt.test_files
    fprate = opt.fprate
    weight = opt.weight
    preset_threshold = opt.preset_threshold

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
    thresh_error_corr = []
    thresh_MAD_corr = []

    dsc_array = np.zeros(3)
    dsc_dev_array = np.zeros(3)
    sub_num = 1

    total_p = 0
    total_n = 0

    n_latent_samples = 25

    # vae_network = VariationalAutoencoder
    # model = VAEModel(vae_network, config, model_name, log_dir)
    # model.initialize()
    # model.load(model_name, step)
    #model.use_vgg19()

    if ".p" in test_filepath:
        test_file = open(test_filepath, "rb")
        import pickle
        test_subs = pickle.load(test_file)[0]
    else:
        test_file = open(test_filepath, "r")
        test_subs = test_file.readlines()

    #test_file = open(test_filepath, "r")
    for test_sub in test_subs:
        if ".p" in test_filepath:
            test_sub = "atlas/"+test_sub.split("T1w/")[-1]
        else:
            test_sub = test_sub[:-1]
        print(test_sub)
        print(sub_num)


    #test_subs = pickle.load(test_file)[0]

    #test_subs = test_file.readlines()

    # run threshold.py to get threshold values
    vae_network = VariationalAutoencoder
    model = VAEModel(vae_network, config, model_name, log_dir)
    model.load(model_name, step)

    if not preset_threshold:
        thr_error, thr_error_corr, thr_MAD = \
        threshold.compute_threshold(fprate, model, img_size, batch_size, renormalized=True,
                                    n_latent_samples = n_latent_samples,
                                    n_random_sub=50)
    else:
        thr_error, thr_error_corr, thr_MAD = preset_threshold
    print(thr_error, thr_error_corr, thr_MAD)

    for test_sub in test_subs:
        # if '031930_t1w_deface_stx' not in test_sub:
        #    continue
        #if 'Case02' in test_sub:
        #    test_sub = test_sub
        #else:
        #    continue
        #test_sub = "brats/"+test_sub.split("mni/")[-1]
        print(test_sub)
        img_filename = test_sub.split("\n")[0]
        img_filename = img_filename.replace('/scratch-second/', '/scratch_net/bmicdl01/Data/')
        # test outer loop
        img = nib.load(img_filename).get_data()
        seg = img_filename.split('normalized')[0] + "seg_cropped.nii.gz"
        seg = nib.load(seg).get_data()
        seg[seg != 0] = 1

        mask = img_filename.replace('normalized_cropped', 'mask_cropped_mask')
        mask = nib.load(mask).get_data()

        subject_name = img_filename.split('_normalized')[0]
        image_original_size = img.shape[1]

        image_original_size = 210

        try:
            os.makedirs("tests/"+str(model_name) + '_samples_test_atlas/' + str(subject_name))
        except OSError:
            pass
        vis_dir = "tests/"+str(model_name) + '_samples_test_atlas/' + str(subject_name)

        #fill image length for batching, n_img%64==0
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

        img_filled = resize(img_filled, img_size / image_original_size, method='bilinear')
        mask_filled = resize(mask_filled, img_size / image_original_size, method='nearest')
        img_len = int(len(img_filled)/batch_size)

        #img_len = np.int(512/64)

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

            model.validate(img_batch, 1)
            recons_samples = model.out_mu_test
            prior_residuals = model.residual_output_test
            plot_batch(prior_residuals, vis_dir + '/pred_error_' + str(i) + '.png')
            error_br = np.abs(img_batch - recons_samples)
            decoded_mu = np.zeros(img_batch.shape)
            for s in range(n_latent_samples):
                model.validate(img_batch, 1)
                recons_samples = model.out_mu_test
                decoded_mu += recons_samples

            #stddev_samples = model.out_std_test

            residuals = model.residual_output_test
            #decoded.append(residuals)

            #decoded = np.asarray(decoded).reshape(n_latent_samples, batch_size, img_size, img_size)
            #decoded = np.asarray(decoded).reshape(n_latent_samples, batch_size, img_size, img_size)
            #decoded = np.transpose(decoded, (1, 2, 3, 0))
            decoded_mu = decoded_mu/n_latent_samples

            print(np.sum(mask_batch))
            if np.sum(mask_batch)>300:
                image_restoration = Restoration(img_batch, mask_batch, vae_network, model_name, config)
                image_restoration.reconstructor_vae()
                image_restoration.load(log_dir, model_name, step)
                restored_images, pred_res_restored = \
                    image_restoration.run_map(weight, 800, prior_residuals, decoded_mu)
                #restored_images, pred_res_restored = \
                #    image_restoration.gaussian_approximator(10., 800,
                #                                            prior_residuals)
            else:
                restored_images = img_batch
                pred_res_restored = np.zeros(img_batch.shape)
            # 30000 for tv norm
            # 8000

            error_restored = np.abs(restored_images - img_batch)
            #error_restored = error_restored.max()-error_restored

            error_sub.extend(error_restored)

            plot_batch(recons_samples, vis_dir + '/reconstruct_' + str(i) + '.png')
            plot_batch(prior_residuals,
                       vis_dir + '/pred_error_' + str(i) + '.png')
            plot_batch(pred_res_restored,
                       vis_dir + '/pred_error_restored' + str(i) + '.png')
            plot_batch(error_br, vis_dir + '/error_no-restore_' + str(i) + '.png')
            plot_batch(error_restored, vis_dir + '/error_restored_' + str(i) + '.png')
            plot_batch(restored_images, vis_dir + '/images_restored_' + str(i) + '.png')
            plot_batch(img_batch, vis_dir + '/images_' + str(i) + '.png')


        #len0=512
        plot_batch(np.asarray(error_sub[:len0]), vis_dir + '/error_restored.png')
        #plot_batch(np.asarray(seg), vis_dir + '/seg.png')
        error_sub = np.reshape(error_sub[:len0], [-1, img_size, img_size])
        error_sub = resize(error_sub, image_original_size / img_size, method="bilinear")

        error_sub_m = error_sub[mask == 1].ravel()
        seg_m = seg[mask == 1].ravel()

        ## evaluate AUC for ROC using universal thresholds
        if not len(thresh_error):
            thresh_error = np.concatenate((np.sort(error_sub_m[::100]), [15]))
            error_tprfpr = np.zeros((2, len(thresh_error)))
            #saved_tpr = np.genfromtxt("tpr_error.txt")
            #x = np.arange(0, len(saved_tpr))
            #f = interpolate.interp1d(x, saved_tpr)
            #saved_tpr_interp = f(np.arange(len(thresh_error)))
            #error_tprfpr[0] = saved_tpr_interp

            #saved_fpr = np.genfromtxt("fpr_error.txt")
            #x = np.arange(0, len(saved_fpr))
            #f = interpolate.interp1d(x, saved_fpr)
            #saved_fpr_interp = f(np.arange(len(thresh_error)))
            #error_tprfpr[1] = saved_fpr_interp

        error_tprfpr += compute_tpr_fpr(seg_m, error_sub_m, thresh_error)
        #print(error_sub_m.min(), error_sub_m.max(), np.percentile(error_sub_m,90))

        total_p += np.sum(seg_m == 1)
        total_n += np.sum(seg_m == 0)

        tpr_error = error_tprfpr[0] / total_p
        fpr_error = error_tprfpr[1] / total_n

        # tpr_error_corr = error_corr_tprfpr[0] / total_p
        # fpr_error_corr = error_corr_tprfpr[1] / total_n

        # tpr_MAD = MAD_corr_tprfpr[0] / total_p
        # fpr_MAD = MAD_corr_tprfpr[1] / total_n

        auc_error = 1. + np.trapz(fpr_error, tpr_error)
        # auc_error_corr = 1. + np.trapz(fpr_error_corr, tpr_error_corr)
        # auc_MAD = 1. + np.trapz(fpr_MAD, tpr_MAD)
        #
        # total_p += np.sum(seg_m == 1)
        # total_n += np.sum(seg_m == 0)

        #auc_error = 1. + np.trapz(fpr_error, tpr_error)

        print(test_sub)
        #np.savetxt("tpr_error_l1.txt",error_tprfpr[0])
        #np.savetxt("fpr_error_l1.txt",error_tprfpr[1])
        print(auc_error, total_p, total_n)

        dsc_error_001 = dsc(error_sub_m > thr_error,  seg_m)
        dsc_error_005 = dsc(error_sub_m > thr_error_corr, seg_m)
        dsc_error_0005 = dsc(error_sub_m > thr_MAD, seg_m)
        print(dsc_error_001, dsc_error_005, dsc_error_0005)
        #dsc_error = dsc(error_sub_m > 0.75, seg_m)
        #print(dsc_error)
        #dsc_error = dsc(error_sub_m > 0.85, seg_m)
        #print(dsc_error)
        #0.65
        dsc_array += np.array([dsc_error_001,dsc_error_005,dsc_error_0005 ])
        print(weight)
        #print(dsc_error)
        print("avg dsc error:{}, {}, {}".format(
            dsc_array[0] / (sub_num * 1.),dsc_array[1] / (sub_num * 1.),dsc_array[2] / (sub_num * 1.)))
        sub_num += 1



