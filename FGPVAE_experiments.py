import argparse
import time
import pickle
import os
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

from utils import make_checkpoint_folder, pandas_res_saver, plot_mnist, import_rotated_mnist, print_trainable_vars
from VAE_utils import mnistVAE
from FGPVAE_model import FGP, forward_pass_FGPVAE_rotated_mnist, predict_FGPVAE_rotated_mnist, \
                         extrapolate_experiment_eval_data, latent_samples_FGPVAE

tfd = tfp.distributions
tfk = tfp.math.psd_kernels


def run_experiment_rotated_mnist_FGPVAE(args):
    """
    Function with tensorflow graph and session for FGPVAE experiments on rotated MNIST data.
    For description of FGPVAE see FGPVAE.tex

    :param args:
    :return:

    :param args:
    :return:
    """

    # define some constants
    n = len(args.dataset)
    N_train = n * np.floor(4050 / args.N_t) * args.N_t
    N_eval = n * 640

    assert not (args.object_prior_corr and args.extrapolate_experiment), \
        "When using correlated object priors, can not do extrapolation experiment!"
    assert args.batch_size % args.N_t == 0

    if args.save:
        # Make a folder to save everything
        extra = args.elbo + "_" + str(args.beta)
        chkpnt_dir = make_checkpoint_folder(args.base_dir, args.expid, extra)
        pic_folder = chkpnt_dir + "pics/"
        res_file = chkpnt_dir + "res/ELBO_pandas"
        print("\nCheckpoint Directory:\n" + str(chkpnt_dir) + "\n")

        json.dump(vars(args), open(chkpnt_dir + "/args.json", "wt"))

    # Init plots
    if args.show_pics:
        plt.ion()

    graph = tf.Graph()
    with graph.as_default():

        # ====================== 1) import data ======================
        # for FGPVAE not shuffled data is always used
        ending = args.dataset + "_not_shuffled.p"

        iterator, training_init_op, eval_init_op, test_init_op, train_data_dict, eval_data_dict, test_data_dict, \
        eval_batch_size_placeholder, test_batch_size_placeholder = import_rotated_mnist(args.mnist_data_path,
                                                                                        ending=ending,
                                                                                        batch_size=args.batch_size,
                                                                                        digits=args.dataset,
                                                                                        N_t=args.N_t)

        if args.extrapolate_experiment:
            observed_images_extra, observed_aux_data_extra, \
            test_images_extra, test_aux_data_extra = extrapolate_experiment_eval_data(mnist_path=args.mnist_data_path,
                                                                                      digit=args.dataset, N_t=args.N_t)

        # get the batch
        input_batch = iterator.get_next()

        # ====================== 2) build ELBO graph ======================

        # init VAE object
        VAE = mnistVAE(L=args.L)
        beta = tf.compat.v1.placeholder(dtype=tf.float64, shape=())

        # init GP object
        if args.ov_joint:
            object_vectors_init = pickle.load(open(args.mnist_data_path + 'pca_ov_init{}.p'.format(args.dataset), 'rb'))
        else:
            object_vectors_init = None

        GP = FGP(init_amplitude=1.0, init_length_scale=1.0, GP_joint=args.GP_joint, L_w=args.L_w,
                  object_vectors_init=object_vectors_init, object_prior_corr=args.object_prior_corr,
                  K_obj_normalize=args.object_kernel_normalize)

        # forward pass
        N_t = tf.compat.v1.placeholder(dtype=tf.int64, shape=())

        C_ma_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
        lagrange_mult_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
        alpha_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=())

        elbo, recon_loss, elbo_kl_part, p_m, p_v, \
        qnet_mu, qnet_var, recon_images, latent_samples, \
        C_ma, lagrange_mult = forward_pass_FGPVAE_rotated_mnist(input_batch,
                                                                 beta=beta,
                                                                 vae=VAE,
                                                                 GP=GP,
                                                                 N_t=N_t,
                                                                 clipping_qs=args.clip_qs,
                                                                 bayes_reg_view=args.bayes_reg_view,
                                                                 omit_C_tilde=args.omit_C_tilde,
                                                                 C_ma=C_ma_placeholder,
                                                                 lagrange_mult=lagrange_mult_placeholder,
                                                                 alpha=alpha_placeholder,
                                                                 kappa=np.sqrt(args.kappa_squared),
                                                                 GECO=args.GECO)

        # prediction
        # TODO: add support for batching for prediction pipeline
        #  (tf.where, tf.boolean_mask, tf.gather etc. to select train images belonging to ids in the test batch)
        train_images_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 28, 28, 1))
        train_aux_data_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 10))
        test_images_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 28, 28, 1))
        test_aux_data_placeholder = tf.compat.v1.placeholder(dtype=tf.float64, shape=(None, 10))
        recon_images_test, recon_loss_test = predict_FGPVAE_rotated_mnist(test_images_placeholder,
                                                                           test_aux_data_placeholder,
                                                                           train_images_placeholder,
                                                                           train_aux_data_placeholder,
                                                                           vae=VAE, GP=GP, N_t=args.N_t,
                                                                           clipping_qs=args.clip_qs,
                                                                           bayes_reg_view=args.bayes_reg_view,
                                                                           omit_C_tilde=args.omit_C_tilde)
        if args.save_latents:
            latent_samples_full = latent_samples_FGPVAE(train_images_placeholder, train_aux_data_placeholder,
                                                         vae=VAE, GP=GP, N_t=N_t, clipping_qs=args.clip_qs)

        # ====================== 3) optimizer ops ======================

        global_step = tf.Variable(0, name='global_step', trainable=False)
        lr = tf.compat.v1.placeholder(dtype=tf.float64, shape=())
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        if args.GECO:
            gradients = tf.gradients(elbo, train_vars)
        else:
            # minimizing the negative elbo!
            gradients = tf.gradients(-elbo, train_vars)

        optim_step = optimizer.apply_gradients(grads_and_vars=zip(gradients, train_vars),
                                               global_step=global_step)

        # ====================== 4) Pandas saver ======================
        if args.save:
            res_vars = [global_step,
                        tf.reduce_sum(elbo) / N_eval,
                        tf.reduce_sum(recon_loss) / N_eval,
                        tf.reduce_sum(elbo_kl_part) / N_eval,
                        tf.math.reduce_min(qnet_var),
                        tf.math.reduce_max(qnet_var),
                        tf.math.reduce_min(p_v),
                        tf.math.reduce_max(p_v),
                        tf.math.reduce_min(qnet_mu),
                        tf.math.reduce_max(qnet_mu),
                        tf.math.reduce_min(p_m),
                        tf.math.reduce_max(p_m),
                        latent_samples,
                        qnet_var,
                        C_ma,
                        lagrange_mult]

            res_names = ["step",
                         "ELBO",
                         "recon loss",
                         "KL term",
                         "min qs_var",
                         "max qs_var",
                         "min q_var",
                         "max q_var",
                         'min qs_mean',
                         'max qs_mean',
                         'min q_mean',
                         'max q_mean',
                         'latent_samples',
                         'full qs_var',
                         "C_ma",
                         "lagrange_mult"]

            res_saver = pandas_res_saver(res_file, res_names)

        # ====================== 5) print and init trainable params ======================

        print_trainable_vars(train_vars)

        init_op = tf.global_variables_initializer()

        # ====================== 6) saver and GPU ======================

        if args.save_model_weights:
            saver = tf.compat.v1.train.Saver(max_to_keep=3)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.ram)

        if args.object_prior_corr:
            N_print = 170
        else:
            N_print = 10

        # ====================== 7) tf.session ======================

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

            sess.run(init_op)

            first_step = True  # switch for initizalition of GECO algorithm
            C_ma_ = 0.0
            lagrange_mult_ = 1.0

            start_time = time.time()
            cgen_test_set_MSE = []
            for epoch in range(args.nr_epochs):

                # handcrafted learning rate and beta schedules
                if epoch < args.beta_schedule_switch:
                    beta_main = args.beta
                    lr_main = args.lr
                else:
                    # beta_main = args.beta
                    beta_main = args.beta / 10
                    lr_main = args.lr / 10

                # 7.1) train for one epoch
                sess.run(training_init_op)
                elbos, losses = [], []
                start_time_epoch = time.time()
                while True:
                    try:
                        if first_step:
                            alpha = 0.0
                        else:
                            alpha = args.alpha
                        _, g_s_, elbo_, C_ma_, lagrange_mult_, recon_loss_ = sess.run([optim_step, global_step,
                                                                                       elbo, C_ma, lagrange_mult,
                                                                                       recon_loss],
                                                                                      {beta: beta_main, lr: lr_main,
                                                                                       alpha_placeholder: alpha,
                                                                                       C_ma_placeholder: C_ma_,
                                                                                       lagrange_mult_placeholder: lagrange_mult_,
                                                                                       N_t: args.N_t})
                        elbos.append(elbo_)
                        losses.append(recon_loss_)
                        first_step = False  # switch for initizalition of GECO algorithm
                    except tf.errors.OutOfRangeError:
                        if (epoch + 1) % N_print == 0:
                            print('Epoch {}, mean ELBO: {}'.format(epoch, np.sum(elbos) / N_train))
                            MSE = np.sum(losses) / N_train
                            print('MSE loss on train set for epoch {} : {}'.format(epoch, MSE))
                            end_time_epoch = time.time()
                            print("Time elapsed for epoch {}: {}".format(epoch, end_time_epoch - start_time_epoch))
                        break

                # 7.2) calculate performance metrics on eval set
                if args.save and (epoch + 1) % N_print == 0:
                    losses, elbos = [], []
                    sess.run(eval_init_op,
                             {eval_batch_size_placeholder: 240})  # since eval batch_size needs to be divisibile by 16
                    while True:
                        try:
                            recon_loss_, elbo_ = sess.run([recon_loss, elbo], {beta: beta_main,
                                                                               N_t: 16,
                                                                               alpha_placeholder: args.alpha,
                                                                               C_ma_placeholder: C_ma_,
                                                                               lagrange_mult_placeholder: lagrange_mult_})
                            losses.append(recon_loss_)
                            elbos.append(elbo_)
                        except tf.errors.OutOfRangeError:
                            print('MSE loss on eval set for epoch {} : {}'.format(epoch, np.sum(losses) / N_eval))
                            break

                # 7.3) save diagnostics metrics to Pandas df
                if args.save and (epoch + 1) % N_print == 0:
                    sess.run(eval_init_op, {eval_batch_size_placeholder: len(eval_data_dict['images'])})
                    new_res = sess.run(res_vars, {beta: args.beta, N_t: 16,
                                                  alpha_placeholder: args.alpha,
                                                  C_ma_placeholder: C_ma_,
                                                  lagrange_mult_placeholder: lagrange_mult_})
                    res_saver(new_res, 1)

                # 7.4) calculate loss on test set and visualize reconstructed images
                if (epoch + 1) % N_print == 0:
                    # 7.4.1) test set: conditional generation
                    n_img = len(args.dataset) * 270 * args.N_t  # number of train images that belong to test ids
                    recon_loss_test_, recon_images_test_ = sess.run([recon_loss_test, recon_images_test],
                                                                    {train_images_placeholder: train_data_dict[
                                                                                                   'images'][:n_img, :],
                                                                     test_images_placeholder: test_data_dict['images'],
                                                                     train_aux_data_placeholder: train_data_dict[
                                                                                                     'aux_data'][:n_img,
                                                                                                 :],
                                                                     test_aux_data_placeholder: test_data_dict[
                                                                         'aux_data']})

                    cgen_test_set_MSE.append((epoch, recon_loss_test_))
                    print(
                        "Conditional generation MSE loss on test set for epoch {}: {}".format(epoch, recon_loss_test_))
                    plot_mnist(test_data_dict['images'], recon_images_test_,
                               title="Epoch: {}. CGEN MSE test set:{}".format(epoch + 1, round(recon_loss_test_, 4)))

                    if args.show_pics:
                        plt.show()
                        plt.pause(0.01)
                    if args.save:
                        plt.savefig(pic_folder + str(g_s_) + "_cgen_test_set.png")
                        with open(pic_folder + "test_metrics.txt", "a") as f:
                            f.write("{},{}\n".format(epoch + 1, round(recon_loss_test_, 4)))

                    # 7.4.2) extrapolate experiment (using validation dataset)
                    if args.extrapolate_experiment:
                        recon_loss_test_, recon_images_test_extrapolate = sess.run([recon_loss_test, recon_images_test],
                                                                                   {
                                                                                       train_images_placeholder: observed_images_extra,
                                                                                       test_images_placeholder: test_images_extra,
                                                                                       train_aux_data_placeholder: observed_aux_data_extra,
                                                                                       test_aux_data_placeholder: test_aux_data_extra})

                        print(
                            "Conditional generation MSE loss for extrapolate experiment for epoch {}: {}".format(epoch,
                                                                                                                 recon_loss_test_))
                        plot_mnist(test_images_extra, recon_images_test_extrapolate,
                                   title="Epoch: {}. CGEN MSE extrapolate experiment:{}".format(epoch + 1,
                                                                                                round(recon_loss_test_,
                                                                                                      4)))

                        if args.show_pics:
                            plt.show()
                            plt.pause(0.01)
                        if args.save:
                            plt.savefig(pic_folder + str(g_s_) + "_cgen_extra_exp.png")
                            with open(pic_folder + "test_metrics.txt", "a") as f:
                                f.write("{},{}\n".format(epoch + 1, round(recon_loss_test_, 4)))

                    # save model weights
                    if args.save and args.save_model_weights:
                        saver.save(sess, chkpnt_dir + "model", global_step=g_s_)

            # log running time
            end_time = time.time()
            print("Running time for {} epochs: {}".format(args.nr_epochs, round(end_time - start_time, 2)))

            # report best test set cgen MSE achieved throughout training
            best_cgen_MSE = sorted(cgen_test_set_MSE, key=lambda x: x[1])[0]
            print("Best cgen MSE on test set throughout training at epoch {}: {}".format(best_cgen_MSE[0],
                                                                                         best_cgen_MSE[1]))

            # save images from conditional generation
            if args.save:
                with open(chkpnt_dir + '/cgen_images.p', 'wb') as test_pickle:
                    pickle.dump(recon_images_test_, test_pickle)

            # save latents
            if args.save_latents:
                latent_samples_full_ = sess.run(latent_samples_full,
                                                {train_images_placeholder: train_data_dict['images'],
                                                 train_aux_data_placeholder: train_data_dict['aux_data'],
                                                 N_t: args.N_t})
                with open(chkpnt_dir + '/latents_train_full.p', 'wb') as pickle_latents:
                    pickle.dump((latent_samples_full_, train_data_dict['aux_data'], train_data_dict['images']),
                                pickle_latents)


if __name__=="__main__":

    default_base_dir = os.getcwd()

    # =============== parser rotated MNIST data ===============
    parser_mnist = argparse.ArgumentParser(description='Train SVGPVAE or FGPVAE for rotated MNIST data.')
    parser_mnist.add_argument('--expid', type=str, default="debug_MNIST", help='give this experiment a name')
    parser_mnist.add_argument('--base_dir', type=str, default=default_base_dir,
                              help='folder within a new dir is made for each run')

    parser_mnist.add_argument('--mnist_data_path', type=str, default='MNIST data/',
                              help='Path where rotated MNIST data is stored.')
    parser_mnist.add_argument('--batch_size', type=int, default=220)
    parser_mnist.add_argument('--nr_epochs', type=int, default=1000)
    parser_mnist.add_argument('--beta', type=float, default=0.001)
    parser_mnist.add_argument('--nr_inducing_points', type=float, default=5, help="Number of object vectors per angle.")
    parser_mnist.add_argument('--save', action="store_true", help='Save model metrics in Pandas df as well as images.')
    parser_mnist.add_argument('--GP_joint', action="store_true", help='GP hyperparams joint optimization.')
    parser_mnist.add_argument('--ov_joint', action="store_true", help='Object vectors joint optimization.')
    parser_mnist.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam optimizer.')
    parser_mnist.add_argument('--not_shuffled', action="store_true", help='Do not shuffle train and test data.')
    parser_mnist.add_argument('--save_model_weights', action="store_true",
                              help='Save model weights. For debug purposes.')
    parser_mnist.add_argument('--dataset', type=str, choices=['3', '36', '13679'], default='3')
    parser_mnist.add_argument('--show_pics', action="store_true", help='Show images during training.')
    parser_mnist.add_argument('--beta_schedule_switch', type=int, default=1000)
    parser_mnist.add_argument('--opt_regime', type=str, default=['joint-120'], nargs="+")
    parser_mnist.add_argument('--L', type=int, default=16, help="Nr. of latent channels")
    parser_mnist.add_argument('--clip_qs', action="store_true", help='Clip variance of inference network.')
    parser_mnist.add_argument('--ram', type=float, default=1.0, help='fraction of GPU ram to use')
    parser_mnist.add_argument('--test_set_metrics', action='store_true',
                              help='Calculate metrics on test data. If false, metrics are calculated on eval data.')
    parser_mnist.add_argument('--GECO', action='store_true', help='Use GECO algorithm for training.')
    parser_mnist.add_argument('--alpha', type=float, default=0.99, help='Moving average parameter for GECO.')
    parser_mnist.add_argument('--kappa_squared', type=float, default=0.033, help='Constraint parameter for GECO.')
    parser_mnist.add_argument('--object_kernel_normalize', action='store_true', help='Normalize object (linear) kernel.')
    parser_mnist.add_argument('--save_latents', action='store_true',
                              help='Save Z . For t-SNE plots :)')

    parser_mnist.add_argument('--L_w', type=int, default=8, help="Nr. of latent channels for feature w")
    parser_mnist.add_argument('--extrapolate_experiment', action="store_true", help='Perform extrapolation experiment.')
    parser_mnist.add_argument('--object_prior_corr', action="store_true",
                              help='Use correlated object prior in FGPVAE.')
    parser_mnist.add_argument('--bayes_reg_view', action="store_true",
                              help='Use Bayesian regression view in linear kernel for FGPVAE --object_prior_corr.')
    parser_mnist.add_argument('--omit_C_tilde', action="store_true",
                              help='Omit C_tilde terms in FGPVAE --object_prior_corr and modify CE term in ELBO instead.')
    parser_mnist.add_argument('--N_t', type=int, default=11,
                              help='How many angels in train set for each image in test set (since FGPVAE implementation is based on not_shuffled data).')

    args_mnist = parser_mnist.parse_args()

    run_experiment_rotated_mnist_FGPVAE(args_mnist)
