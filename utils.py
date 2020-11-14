import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops import math_ops as tfmath_ops
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt
import glob
from matplotlib.patches import Ellipse
import shutil
import pandas as pd
import pickle
import time
import subprocess as sp
import math


import random
from sklearn.decomposition import PCA
from scipy import ndimage
import scipy
import seaborn as sns

tfk = tfp.math.psd_kernels


def make_checkpoint_folder(base_dir, expid=None, extra=""):
    """
    Makes a folder and sub folders for pics and results
    Args:
        base_dir: the root directory where new folder will be made
        expid: optional extra sub dir inside base_dir
    """

    # make a "root" dir to store all checkpoints
    # homedir = os.getenv("HOME")
    # base_dir = homedir+"/GPVAE_checkpoints/"

    if expid is not None:
        base_dir = base_dir + "/" + expid + "/"

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # now make a unique folder inside the root for this experiments
    filenum = str(len(os.listdir(base_dir))) + "_"+extra+"__on__"

    T = dt.now()

    filetime = str(T.day)+"_"+str(T.month)+"_"+str(T.year) + "__at__"
    filetime += str(T.hour)+"_"+str(T.minute)+"_"+str(T.second)

    # main folder
    checkpoint_folder = base_dir + filenum + filetime
    os.makedirs(checkpoint_folder)

    # pictures folder
    pic_folder = checkpoint_folder + "/pics/"
    os.makedirs(pic_folder)

    # pickled results files
    res_folder = checkpoint_folder + "/res/"
    os.makedirs(res_folder)

    # source code
    src_folder = checkpoint_folder + "/sourcecode/"
    os.makedirs(src_folder)
    old_src_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    src_files = os.listdir(old_src_dir)
    print("\n\nCopying source Code to "+src_folder)
    for f in src_files:
        if ".py" in f:
            src_file = old_src_dir + f
            shutil.copy2(src_file, src_folder)
            print(src_file)
    print("\n")

    # predictions folder, for plotting purposes
    preds_folder = checkpoint_folder + "/preds/"
    os.makedirs(preds_folder)

    
    return checkpoint_folder + "/"


class pandas_res_saver:
    """
    Takes a file and a list of col names to initialise a
    pandas array. Then accepts extra rows to be added
    and occasionally written to disc.
    """
    def __init__(self, res_file, colnames):
        # reload old results frame
        if os.path.exists(res_file):
            if list(pd.read_pickle(res_file).columns)==colnames:
                print("res_file: recovered ")
                self.data = pd.read_pickle(res_file)
                self.res_file = res_file
            else:
                print("res_file: old exists but not same, making new ")
                self.res_file = res_file + "_" + str(time.time())
                self.data = pd.DataFrame(columns=colnames)
        else:
            print("res_file: new")
            self.res_file = res_file
            self.data = pd.DataFrame(columns=colnames)
            
        self.ncols = len(colnames)
        self.colnames = colnames
    
    def __call__(self, new_data, n_steps=10):
        new_data = np.asarray(new_data).reshape((-1, self.ncols))
        new_data = pd.DataFrame(new_data, columns=self.colnames)
        self.data = pd.concat([self.data, new_data])

        if self.data.shape[0]%n_steps == 0:
            self.data.to_pickle(self.res_file)
            print("Saved results to file: "+self.res_file)


def gauss_cross_entropy(mu1, var1, mu2, var2):
    """
    Computes the element-wise cross entropy
    Given q(z) ~ N(z| mu1, var1)
    returns E_q[ log N(z| mu2, var2) ]
    args:
        mu1:  mean of expectation (batch, tmax, 2) tf variable
        var1: var  of expectation (batch, tmax, 2) tf variable
        mu2:  mean of integrand (batch, tmax, 2) tf variable
        var2: var of integrand (batch, tmax, 2) tf variable
    returns:
        cross_entropy: (batch, tmax, 2) tf variable
    """

    term0 = 1.8378770664093453  # log(2*pi)
    term1 = tf.log(var2)
    term2 = (var1 + mu1 ** 2 - 2 * mu1 * mu2 + mu2 ** 2) / var2

    cross_entropy = -0.5 * (term0 + term1 + term2)

    return cross_entropy
    

def generate_rotated_MNIST(save_path, N=400, nr_angles=16, valid_set_size=0.1, drop_rate=0.25, digits=[3, 6],
                           latent_dim_object_vector=8, shuffle_data=True, seed=0):
    """
    Generate rotated MNIST data from Casale's paper.

    Saves train, validation and test sets as pickle files.
    Each dataset is a Pyhton dictionary with keys: ['images', 'auxiliary data'].
    Auxiliary data consists of image id, rotation angle and PCA embedding vector.

    :param save_path: path for saving the generated data
    :param N: number of MNIST images of specified digits to use
    :param nr_angles: number of angles between [0, 2pi) considered
    :param valid_set_size: size of validation set
    :param drop_rate: how much images to drop
    :param digit: which digit to consider
    :param shuffle_data: whether or not to shuffle data. Might be important since if we pass
        all angles of the same digit in same batch, kernel matrices carry more information that model could exploit.
        Note that for Michael's extrapolatingGPVAE idea, data should not be shuffled, since there independent GPs are
        fitted for each image.
    :param latent_dim_object_vector: dimension of latent dimension of object vectors
    :param seed: random seed, for reproducibility
    """

    random.seed(seed)
    angles = np.linspace(0, 360, nr_angles + 1)[:-1]

    # load MNIST data
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train = x_train[..., np.newaxis] / 255.0

    # TODO: should MNIST images be binarized here?

    # filter out images with correct digit
    digits_df = []
    for digit in digits:
        x_train_digit = x_train[(y_train == digit)]
        print('Number of images with digit {}: {}'.format(digit, len(x_train_digit)))

        # subsample N images
        indices = random.sample(list(range(x_train_digit.shape[0])), N)
        digits_df.append(x_train_digit[indices, :, :, 0])  # (N, 28, 28)

    x_train = np.concatenate(digits_df)

    # PCA
    pca_df = x_train.copy().reshape((x_train.shape[0], -1))
    pca = PCA(n_components=latent_dim_object_vector)
    pca_df = pca.fit_transform(pca_df)
    print("Explained variance ratio PCA: {}".format(pca.explained_variance_ratio_))

    # save pca_df to pickle (for init of object vectors)
    digit_ending = "".join([str(x) for x in digits])
    with open(save_path + 'pca_ov_init{}_{}.p'.format(digit_ending, latent_dim_object_vector), 'wb') as ov_init_pickle:
        pickle.dump(pca_df, ov_init_pickle)

    # rotate images
    def rotate_image(image, image_id, angles, pca_embedding):
        images = []

        aux_data = np.array([tuple([image_id, math.radians(angle)] + list(pca_embedding)) for angle in angles])

        for i in range(len(angles)):
            images.append(ndimage.rotate(image, angles[i], reshape=False))

        images = np.stack(images)
        images = images[..., np.newaxis]

        return images, aux_data

    images, aux_data = [], []

    assert len(digits) * N == x_train.shape[0]
    for i in range(len(digits) * N):
        images_rot, aux_data_i = rotate_image(x_train[i, :, :], i, angles, pca_df[i, :].copy())
        images.append(images_rot)
        aux_data.append(aux_data_i)

    images = np.concatenate(images)  # (N * len(angles), 28, 28, 1)
    aux_data = np.concatenate(aux_data)  # (N * len(angles), 10)

    # train/test and eval split
    images_, aux_data_, eval_images_, eval_aux_data_ = [], [], [], []
    N_digit = int(len(images) / len(digits))
    N_eval = int(N_digit * (1 - valid_set_size))
    for i in range(len(digits)):
        images_.append(images[i * N_digit:i * N_digit + N_eval])
        aux_data_.append(aux_data[i * N_digit:i * N_digit + N_eval])
        eval_images_.append(images[i * N_digit + N_eval:(i + 1) * N_digit])
        eval_aux_data_.append(aux_data[i * N_digit + N_eval:(i + 1) * N_digit])

    images, aux_data, eval_images, eval_aux_data = np.concatenate(images_), np.concatenate(aux_data_), \
                                                   np.concatenate(eval_images_), np.concatenate(eval_aux_data_)

    # shuffle eval data
    if shuffle_data:
        eval_idx = random.sample(list(range(len(eval_images))), len(eval_images))
        eval_images, eval_aux_data = eval_images[eval_idx], eval_aux_data[eval_idx]

    # train and test split
    test_angle = random.sample(list(angles), 1)[0]
    mask = (aux_data[:, 1] == math.radians(test_angle))
    train_images, train_aux_data, test_images, test_aux_data = images[~mask], aux_data[~mask], \
                                                               images[mask], aux_data[mask]
    print("Test angle: {}".format(test_angle))

    # drop some images
    if shuffle_data:
        idx_train = random.sample(list(range(len(train_images))), int(len(train_images) * (1 - drop_rate)))
        idx_test = random.sample(list(range(len(test_images))), int(len(test_images) * (1 - drop_rate)))
    else:
        idx_train = list(range(int(len(train_images) * (1 - drop_rate))))
        idx_test = list(range(int(len(test_images) * (1 - drop_rate))))

        idx_train_not_in_test = list(range(int(len(train_images) * (1 - drop_rate)), len(train_images)))
        train_not_in_test_images = train_images[idx_train_not_in_test]
        train_not_in_test_aux_data = train_aux_data[idx_train_not_in_test]

    train_images, train_aux_data = train_images[idx_train], train_aux_data[idx_train]
    test_images, test_aux_data = test_images[idx_test], test_aux_data[idx_test]

    print('Size of training data: {}'.format(len(train_images)))
    print('Size of validation data: {}'.format(len(eval_images)))
    print('Size of test data: {}'.format(len(test_images)))

    if not shuffle_data:
        print('Size of training data without test ids: {}'.format(len(train_not_in_test_images)))

    # save to pickle files
    train_dict = {'images': train_images, 'aux_data': train_aux_data}
    eval_dict = {'images': eval_images, 'aux_data': eval_aux_data}
    test_dict = {'images': test_images, 'aux_data': test_aux_data}

    if not shuffle_data:
        train_not_in_test_dict = {'images': train_not_in_test_images, 'aux_data': train_not_in_test_aux_data}

    ending = "_not_shuffled_{}.p".format(latent_dim_object_vector) if not shuffle_data else "_{}.p".format(latent_dim_object_vector)
    ending = digit_ending + ending
    print(ending)

    with open(save_path + 'train_data' + ending, 'wb') as train_pickle:
        pickle.dump(train_dict, train_pickle)
    with open(save_path + 'eval_data' + ending, 'wb') as eval_pickle:
        pickle.dump(eval_dict, eval_pickle)
    with open(save_path + 'test_data' + ending, 'wb') as test_pickle:
        pickle.dump(test_dict, test_pickle)

    if not shuffle_data:
        with open(save_path + 'train_not_in_test_data' + ending, 'wb') as train_pickle:
            pickle.dump(train_not_in_test_dict, train_pickle)


def plot_mnist(arr, recon_arr, title, nr_images=8, seed=0):
    """

    :param arr:
    :param recon_arr:
    :param title:
    :param nr_images:
    :param seed:
    :return:
    """
    random.seed(seed)
    assert nr_images % 8 == 0

    indices = random.sample(list(range(len(arr))), nr_images)
    plt.figure(figsize=(10, 10*int(nr_images/8)))
    plt.suptitle(title)
    for i in range(int(nr_images*2)):
        plt.subplot(int(nr_images / 2), 4, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if i % 2 == 0:
            plt.imshow(arr[indices[i // 2]][:, :, 0], cmap='gray')
            plt.xlabel("Ground truth, id: {}".format(indices[i // 2]))
        else:
            plt.imshow(recon_arr[indices[i // 2]][:, :, 0], cmap='gray')
            plt.xlabel("Recon image, id: {}".format(indices[i // 2]))
    # plt.tight_layout()
    plt.draw()


def visualize_kernel_matrices(aux_data_arr, batch_size=32, N=1, K_obj_normalized=True,
                              amplitude=1.0, length_scale=1.0):
    """
    Visualize heatmaps of kernel matrices.

    :param aux_data_arr:
    :param batch_size:
    :param N: number of batches to visualize
    :param K_obj_normalized: whether or not to normalize (between -1 and 1) object kernel matrix (linear kernel)
    :param amplitude:
    :param length_scale:
    """

    # define kernels
    kernel_view = tfk.ExpSinSquared(amplitude=amplitude, length_scale=length_scale, period=2 * np.pi)
    kernel_object = tfk.Linear()
    x = tf.placeholder(dtype=tf.float32)
    y = tf.placeholder(dtype=tf.float32)
    z = tf.placeholder(dtype=tf.float32)
    w = tf.placeholder(dtype=tf.float32)
    K_view = kernel_view.matrix(tf.expand_dims(x, axis=1), tf.expand_dims(y, axis=1))
    K_obj = kernel_object.matrix(z, w)
    if K_obj_normalized:
        obj_norm = 1 / tf.matmul(tf.math.reduce_euclidean_norm(z, axis=1, keepdims=True),
                                 tf.transpose(tf.math.reduce_euclidean_norm(z, axis=1, keepdims=True), perm=[1, 0]))
        K_obj = K_obj * obj_norm
    K_prod = K_view * K_obj

    # util function for heatmaps
    def heatmap(ax_, arr, title, vmin=0, vmax=1):
        ax = sns.heatmap(arr, vmin=vmin, vmax=vmax, center=0,
                         cmap=sns.diverging_palette(20, 220, n=200),
                         square=True, ax=ax_)
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=45,
                           horizontalalignment='right')
        ax.set_title(title);

    for i in range(N):
        # generate kernel matrices
        batch = aux_data_arr[i * batch_size:(i + 1) * batch_size]
        with tf.Session() as sess:
            K_view_, K_obj_, K_prod_ = sess.run([K_view, K_obj, K_prod],
                                                {x: batch[:, 1], y: batch[:, 1], z: batch[:, 2:], w: batch[:, 2:]})
        # plot kernel matrices
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        heatmap(axes[0], K_view_, "View kernel. Batch: {}. Det: {}".format(i + 1, np.linalg.det(K_view_)))
        heatmap(axes[1], K_obj_, "Object kernel. Batch: {}. Det: {}".format(i + 1, np.linalg.det(K_obj_)), vmin=-1)
        heatmap(axes[2], K_prod_, "Product kernel. Batch: {}. Det: {}".format(i + 1, np.linalg.det(K_prod_)), vmin=-1)
    plt.show()


def import_rotated_mnist(MNIST_path, ending, batch_size, digits="3", N_t=None):
    """

    Support for loading of data and batching via tf.data.Dataset API.

    :param MNIST_path:
    :param ending:
    :param batch_size:
    :param N_t: How many angels in train set for each image in test set
                (since reGPVAE implementation is based on not_shuffled data).

    :return:
    """

    # TODO: here we load entire data in the memory. For MNIST that is fine, for larger datasets will have to
    #  implement it in more efficient way

    # train data
    train_data_dict = pickle.load(open(MNIST_path + 'train_data' + ending, 'rb'))
    if N_t is not None:
        flatten = lambda l: [item for sublist in l for item in sublist]
        digit_mask = [True] * N_t + [False] * (15 - N_t)

        mask = [random.sample(digit_mask, len(digit_mask)) for _ in range(int(len(train_data_dict['aux_data'])/15))]
        mask = flatten(mask)
        train_data_dict['images'] = train_data_dict['images'][mask]
        train_data_dict['aux_data'] = train_data_dict['aux_data'][mask]

        # add train images without test angles
        if N_t < 15:
            train_not_in_test_data_dict = pickle.load(open(MNIST_path + 'train_not_in_test_data' + ending, 'rb'))

            n = int(len(digits) * 270 * (15 - N_t) / N_t) * N_t

            mask = [random.sample(digit_mask, len(digit_mask)) for _ in range(int(len(train_not_in_test_data_dict['aux_data']) / 15))]
            mask = flatten(mask)

            train_data_dict['images'] = np.concatenate((train_data_dict['images'],
                                                        train_not_in_test_data_dict['images'][mask][:n, ]), axis=0)
            train_data_dict['aux_data'] = np.concatenate((train_data_dict['aux_data'],
                                                          train_not_in_test_data_dict['aux_data'][mask][:n, ]), axis=0)

    train_data_images = tf.data.Dataset.from_tensor_slices(train_data_dict['images'])
    train_data_aux_data = tf.data.Dataset.from_tensor_slices(train_data_dict['aux_data'])
    train_data = tf.data.Dataset.zip((train_data_images, train_data_aux_data)).batch(batch_size)

    # eval data
    eval_batch_size_placeholder = tf.compat.v1.placeholder(dtype=tf.int64, shape=())
    eval_data_dict = pickle.load(open(MNIST_path + 'eval_data' + ending, 'rb'))
    eval_data_images = tf.data.Dataset.from_tensor_slices(eval_data_dict['images'])
    eval_data_aux_data = tf.data.Dataset.from_tensor_slices(eval_data_dict['aux_data'])
    eval_data = tf.data.Dataset.zip((eval_data_images, eval_data_aux_data)).batch(eval_batch_size_placeholder)

    # test data
    test_batch_size_placeholder = tf.compat.v1.placeholder(dtype=tf.int64, shape=())
    test_data_dict = pickle.load(open(MNIST_path + 'test_data' + ending, 'rb'))
    test_data_images = tf.data.Dataset.from_tensor_slices(test_data_dict['images'])
    test_data_aux_data = tf.data.Dataset.from_tensor_slices(test_data_dict['aux_data'])
    test_data = tf.data.Dataset.zip((test_data_images, test_data_aux_data)).batch(test_batch_size_placeholder)

    # init iterator
    iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    training_init_op = iterator.make_initializer(train_data)
    eval_init_op = iterator.make_initializer(eval_data)
    test_init_op = iterator.make_initializer(test_data)

    return iterator, training_init_op, eval_init_op, test_init_op, \
           train_data_dict, eval_data_dict, test_data_dict, eval_batch_size_placeholder, test_batch_size_placeholder


def print_trainable_vars(vars):
    total_parameters = 0
    print("\n\nTrainable variables:")
    for v in vars:
        print(v)
        shape = v.get_shape()
        var_params = 1
        for dim in shape:
            var_params *= dim.value
        total_parameters += var_params
    print("Number of train params: {}".format(total_parameters))


def latent_samples_VAE_full_train(train_images, vae, clipping_qs=False):
    """
    Get latent samples for training data. For t-SNE plots :)

    :param train_images:
    :param vae:
    :param clipping_qs:
    :return:
    """

    # ENCODER NETWORK
    qnet_mu, qnet_var = vae.encode(train_images)

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-3, 10)

    # SAMPLE
    epsilon = tf.random.normal(shape=tf.shape(qnet_mu), dtype=vae.dtype)
    latent_samples = qnet_mu + epsilon * tf.sqrt(qnet_var)

    return latent_samples


if __name__=="__main__":

    # generate_init_inducing_points("MNIST data/train_data3.p", PCA=False)

    # ============= generating rotated MNIST data =============
    # generate_rotated_MNIST("MNIST data/", digits=[3, 6])
    # generate_rotated_MNIST("MNIST data/", digits=[1, 3, 6, 7, 9])
    # generate_rotated_MNIST("MNIST data/")
    # generate_rotated_MNIST('MNIST data/', shuffle_data=False, digits=[6])
    # generate_rotated_MNIST('MNIST data/', shuffle_data=False, digits=[3])
    # generate_rotated_MNIST('MNIST data/', shuffle_data=False, digits=[3, 6])
    # generate_rotated_MNIST('MNIST data/', shuffle_data=False, digits=[1, 3, 6, 7, 9])
    # generate_rotated_MNIST('MNIST data/', shuffle_data=True, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # generate_rotated_MNIST('MNIST data/', shuffle_data=False, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # generate_rotated_MNIST("MNIST data/", digits=[3], latent_dim_object_vector=4)
    # generate_rotated_MNIST("MNIST data/", digits=[3], latent_dim_object_vector=16)
    # generate_rotated_MNIST("MNIST data/", digits=[3], latent_dim_object_vector=32)
    # generate_rotated_MNIST("MNIST data/", digits=[3], latent_dim_object_vector=64)
    generate_rotated_MNIST("MNIST data/", digits=[3], latent_dim_object_vector=24)

