# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SGE_GPU']
# os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import tensorflow as tf
import yaml
import argparse
from utils.batches import get_camcan_batches, tile, plot_batch
from networks.vae_res_bilinear_conv import VariationalAutoencoder
from models.vae import VAEModel
from preprocess.preprocess import *
# from pdb import set_trace as bp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=0)
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default="logs", help="path to checkpoint to restore")
    parser.add_argument("--train", type=str)
    parser.add_argument("--val", type=str)

    opt = parser.parse_args()

    log_dir = os.path.join("logs")
    try:
        os.mkdir(log_dir)
    except OSError:
        pass

    with open(opt.config) as f:
        config = yaml.load(f)
    model_name = opt.model_name
    train_set_name = opt.train
    val_set_name = opt.val

    epochs = config['lr_decay_end']
    batch_size = config["batch_size"]
    img_size = config["spatial_size"]
    img_shape = 2 * [config["spatial_size"]] + [1]
    data_shape = [batch_size] + img_shape
    init_shape = [config["init_batches"] * batch_size] + img_shape
    box_factor = config["box_factor"]
    data_index = config["data_index"]
    z_dim =config["z_dim"]
    LR = config["lr"]
    image_original_size = 200

    try:
        os.mkdir(os.path.join(model_name + "_samples"))
    except OSError:
        pass

    batches = get_camcan_batches(data_shape, train_set_name, train=True, box_factor=box_factor)
    init_batches = get_camcan_batches(init_shape, data_index, train=True, box_factor=box_factor)
    valid_batches = get_camcan_batches(data_shape, val_set_name, train=False, box_factor=box_factor)

    vae_network = VariationalAutoencoder
    model = VAEModel(vae_network, config, model_name, log_dir)
    model.initialize()

    for ep in range(epochs):
        #weight = ((ep - 5e3) / (np.sqrt(1 + (ep - 5e3) ** 2)) + 1) * 0.5
        weight = 1
        input_images, input_masks = next(batches)[:2]
        input_images = input_images.astype("float32")
        input_masks = input_masks.astype("float32")
        model.train(input_images,weight)

        if ep % 500 == 0:
            print(weight)
            validate_images, masks = next(valid_batches)[:2]
            input_images = input_images.astype("float32")
            masks = masks.astype("float32")
            model.validate(validate_images,weight)
            model.visualize(model_name, ep)
            gen_loss, res_loss, lat_loss = model.sess.run([model.autoencoder_loss,
                                                           model.autoencoder_res_loss,
                                                           model.latent_loss], {model.image_matrix: input_images})
            gen_loss_valid, res_loss_valid, lat_loss_valid = model.sess.run([model.autoencoder_loss_test,
                                                           model.autoencoder_res_loss_test,
                                                           model.latent_loss_test], {model.image_matrix: validate_images})
            print(("epoch %d: train_gen_loss %f train_lat_loss %f train_res_loss %f total train_loss %f") % (
                ep, gen_loss.mean(), lat_loss.mean(), res_loss.mean(), gen_loss.mean()+lat_loss.mean()+res_loss.mean()))

            print(("epoch %d: test_gen_loss %f test_lat_loss %f res_loss %f total loss %f") % (
                ep, gen_loss_valid.mean(), lat_loss_valid.mean(), res_loss.mean(),
                gen_loss_valid.mean()+lat_loss_valid.mean()+res_loss.mean()))
            model.save(model_name, ep)

