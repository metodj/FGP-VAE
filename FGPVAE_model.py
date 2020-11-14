import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pickle
import random

from utils import gauss_cross_entropy

tfk = tfp.math.psd_kernels


def _add_diagonal_jitter(matrix, jitter=1e-6):
    return tf.linalg.set_diag(matrix, tf.linalg.diag_part(matrix) + jitter)


class FGP:

    dtype = np.float64

    def __init__(self, init_amplitude, init_length_scale, GP_joint, L_w,
                 object_vectors_init=None, object_prior_corr=False, K_obj_normalize=False):
        """
        GP class for FGPVAE.

        :param init_amplitude:
        :param init_length_scale:
        :param GP_joint:
        :param L_w: number of local latent channels
        :param object_vectors_init: initizalition for object vectors (GP-LVM)
        :param object_prior_corr: whether or not correlated object priors are used
        :param K_obj_normalize: whether or not to normalize object kernel (linear kernel)
        """

        self.object_prior_corr = object_prior_corr
        self.K_obj_normalize = K_obj_normalize

        if GP_joint:
            self.amplitude = tf.Variable(initial_value=init_amplitude,
                                         name="GP_amplitude", trainable=True, dtype=self.dtype)
            self.length_scale = tf.Variable(initial_value=init_length_scale,
                                            name="GP_length_scale", trainable=True, dtype=self.dtype)
        else:
            self.amplitude = tf.constant(init_amplitude, dtype=self.dtype)
            self.length_scale = tf.constant(init_length_scale, dtype=self.dtype)

        # kernels
        self.kernel_local = tfk.ExpSinSquared(amplitude=self.amplitude, length_scale=self.length_scale, period=2*np.pi)
        self.kernel_global = tfk.Linear()

        # GP-LVM, object vectors
        if object_vectors_init is not None:
            self.object_vectors = tf.Variable(initial_value=object_vectors_init,
                                              name="GP_LVM_object_vectors",
                                              dtype=self.dtype)
        else:
            self.object_vectors = None

        # number of local (views/angles) channels
        self.L_w = L_w

    def build_1d_gp_local(self, X, Y, varY, X_test):
        """
        Fits GP for local latent channels.

        Takes input-output dataset and returns post mean, var, marginal lhood.
        This is standard GP regression with heteroscedastic noise.

        :param X: inputs tensor (batch, npoints)
        :param Y: outputs tensor (batch, npoints)
        :param varY: outputs tensor (batch, npoints)
        :param X_test: (batch, ns) input points to compute post mean + var

        Returns:
            p_m: (batch, ns) post mean at X_test
            p_v: (batch, ns) post var at X_test
            logZ: (batch) marginal lhood of each dataset in batch
        """

        # Prepare all constants
        batch = tf.shape(X)[0]
        n = tf.shape(X)[1]
        ns = tf.shape(X_test)[1]

        # K_x + \sigma_x^*
        K = self.kernel_local.matrix(tf.expand_dims(X, 2),  tf.expand_dims(X, 2))  # (batch, n n)
        K = K + tf.matrix_diag(varY)  # (batch, n, n)
        chol_K = tf.linalg.cholesky(K)  # (batch, n, n)

        # lhood term 1/3
        lhood_pi_term = tf.cast(n, dtype=self.dtype) * np.log(2 * np.pi)

        # lhood term 2/3
        lhood_logdet_term = 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_K)), 1)  # (batch)

        # lhood term 3/3
        Y = tf.expand_dims(Y, 2)
        iKY = tf.cholesky_solve(chol_K, Y)  # (batch, n, 1)
        lh_quad_term = tf.matmul(tf.transpose(Y, (0,2,1)), iKY)  # (batch, 1, 1)
        lh_quad_term = tf.reshape(lh_quad_term, [batch])

        # log P(Y|X) = -1/2 * ( n log(2 pi) + Y inv(K+noise) Y + log det(K+noise))
        gp_lhood = -0.5 * (lhood_pi_term + lh_quad_term + lhood_logdet_term)

        # Compute posterior mean and variances
        Ks = self.kernel_local.matrix(tf.expand_dims(X, 2),  tf.expand_dims(X_test, 2))  # (batch, n, ns)
        Ks_t = tf.transpose(Ks, (0, 2, 1))   # (batch, ns, n)

        # posterior mean
        p_m = tf.matmul(Ks_t, iKY)
        p_m = tf.reshape(p_m, (batch, ns))

        # posterior variance
        iK_Ks = tf.cholesky_solve(chol_K, Ks)  # (batch, n, ns)
        Ks_iK_Ks = tf.reduce_sum(Ks * iK_Ks, axis=1)  # (batch, ns)
        p_v = 1 - Ks_iK_Ks  # (batch, ns)
        p_v = tf.reshape(p_v, (batch, ns))

        return p_m, p_v, gp_lhood, K

    def build_1d_gp_global(self, means, vars):
        """
        Fits GP for global latent channels.

        :param Y: encoder means (batch, npoints)
        :param varY: encoder vars (batch, npoints)

        Returns:
            p_m: (batch) posterior means
            p_v: (batch) post vars
            logZ: (batch) product of Gaussians terms
        """
        n = tf.shape(means)[1]

        sigma_squared_bar = 1 / (tf.reduce_sum(tf.math.reciprocal_no_nan(vars), axis=1) + 1)
        mu_bar = sigma_squared_bar * tf.reduce_sum(means * tf.math.reciprocal_no_nan(vars), axis=1)

        lhood = tf.log(tf.sqrt(sigma_squared_bar)) + 0.5*tf.math.reciprocal_no_nan(sigma_squared_bar)*mu_bar**2 - \
                0.5*tf.cast(n, dtype=self.dtype)*tf.log(2.0*tf.cast(np.pi, dtype=self.dtype)) - \
                tf.reduce_sum(tf.log(tf.sqrt(vars)), axis=1) - 0.5*tf.reduce_sum(tf.math.reciprocal_no_nan(vars)*means**2)

        return mu_bar, sigma_squared_bar, lhood

    @staticmethod
    def preprocess_1d_gp_global_correlated_object_priors(means, vars):
        """

        Product of Gaussians for each global latent channel. See 2.9 in FGPVAE.tex

        N = nr. of digits
        N_t = nr. of angles for digit t

        :param means:  (N, N_t)
        :param vars: (N, N_t)

        Returns:
            bar_means: \Bar{\mu} (1, N,)
            bar_vars: \Bar{\sigma}^2 (1, N,)
            C_tilde: \Tilde{C}  (1, N,)
        """

        N_t = tf.shape(means)[1]
        N_t = tf.cast(N_t, dtype=tf.float64)

        alpha = tf.reduce_sum(tf.math.reciprocal_no_nan(vars), axis=1)
        beta = tf.reduce_sum(means / vars, axis=1)

        bar_means = tf.expand_dims(beta / alpha, 0)  # expand_dims to make it compatible with batching latter on
        bar_vars = tf.expand_dims(1 / alpha, 0)  # expand_dims to make it compatible with batching latter on

        # C_1 = (2.0 * np.pi)**(-0.5 * N_t) * tf.reduce_prod(vars**(-0.5), axis=1)
        C_1 = (2.0 * np.pi) ** (-0.5 * N_t) * tf.reduce_prod(tf.sqrt(tf.math.reciprocal_no_nan(vars)), axis=1)
        C_2 = tf.exp(-0.5*tf.reduce_sum(means**2/vars, axis=1))
        C_3 = tf.exp(0.5*beta**2 / alpha)
        C_4 = tf.sqrt(2*np.pi/alpha)

        C_tilde = tf.expand_dims(C_1*C_2*C_3*C_4, 0)  # expand_dims to make it compatible with batching latter on

        # C_tilde = tf.clip_by_value(C_tilde, 1e-90, 100)
        bar_vars = tf.clip_by_value(bar_vars, 1e-3, 100)

        return bar_means, bar_vars, C_tilde

    def kernel_matrix_correlated_object_priors(self, x, y):
        """
        Computes object kernel matrix in case correlated object priors are used.
        See 2.9 in FGPVAE.tex

        :param x: (1, N, 10)
        :param y: (1, N, 10)
        :param K_obj_normalized: whether or not to normalize (between -1 and 1) object kernel matrix (linear kernel)

        :return: object kernel matrix (1, N, N)
        """

        # unpack auxiliary data
        if self.object_vectors is None:
            x_object, y_object =x[:, :, 2:], y[:, :, 2:]
        else:
            x_object = tf.gather(self.object_vectors, tf.cast(x[:, :, 0], dtype=tf.int64))
            y_object = tf.gather(self.object_vectors, tf.cast(y[:, :, 0], dtype=tf.int64))

        # compute kernel matrix
        object_matrix = self.kernel_global.matrix(x_object, y_object)
        if self.K_obj_normalize:  # normalize object matrix
            obj_norm = 1 / tf.matmul(tf.math.reduce_euclidean_norm(x_object, axis=2, keepdims=True),
                                     tf.transpose(tf.math.reduce_euclidean_norm(y_object, axis=2, keepdims=True),
                                                  perm=[0, 2, 1]))
            object_matrix = object_matrix * obj_norm

        return object_matrix

    def X_matrix(self, x):
        """
        Computes X matrix. We need this function (instead of working directly with X) in order to support GP-LVM vectors
        joint optimization.

        :param x: (1, N, 10)
        :param normalized: whether or not to normalize object vectors (so that every object vector has norm 1)

        :return:
        """

        # unpack auxiliary data
        if self.object_vectors is None:
            x_object = x[:, :, 2:]
        else:
            x_object = tf.gather(self.object_vectors, tf.cast(x[:, :, 0], dtype=tf.int64))

        if self.K_obj_normalize:
            x_object = x_object / tf.math.reduce_euclidean_norm(x_object, axis=2, keepdims=True)

        return x_object

    def build_1d_gp_global_correlated_object_priors(self, X, Y, varY, X_test, C_tilde, omit_C_tilde,
                                                    bayesian_reg_view, EPSILON=1e-6):
        """
        See 2.9 in FGPVAE.tex

        Since using build_1d_gp_global_correlated_object_priors leads to numerical issues,
        we add support for fitting global GP using Bayesian linear regression view.

        :param X: auxiliary data, train points of GP (1, N, 10)
        :param Y: encoded and processed means for train points (1, N)
        :param varY: encoded and processed vars for train points (1, N)
        :param X_test: auxiliary data, test points of GP (1, N_s, 10)
        :param C_tilde: (1, N)
        :param omit_C_tilde: omit C_tilde from derivation and modify cross-entropy term instead
        :param bayesian_reg_view: whether or not to use Bayesian regression view to fit global GP.
        :param EPSILON: for numerical stability in log()

        :return:
        """

        if bayesian_reg_view:

            p = 8  # dimension of object vectors
            N = tf.shape(X)[1]

            # get (and normalize) X and X_test
            X = self.X_matrix(X)  # (1, N, p)
            X_T = tf.transpose(X, (0, 2, 1))  # (1, p, N)
            X_test = self.X_matrix(X_test)  # (1, N_s, p)
            X_test_T = tf.transpose(X_test, (0, 2, 1))  # (1, p, N_s)

            # posterior params
            A = tf.matmul(X_T, tf.matmul(tf.linalg.diag(tf.math.reciprocal_no_nan(varY)), X)) + \
                tf.expand_dims(tf.eye(p, dtype=tf.float64), 0)  # (1, p, p)
            A_inv = tf.linalg.inv(_add_diagonal_jitter(A))  # (1, p, p)
            w_bar = tf.linalg.matvec(A_inv, tf.linalg.matvec(X_T, tf.math.reciprocal_no_nan(varY) * Y))  # (1, p)

            p_m = tf.linalg.matvec(X_test, w_bar)  # (1, N)
            p_v = tf.linalg.diag_part(tf.matmul(X_test, tf.matmul(A_inv, X_test_T)))  # (1, N)
            p_v = tf.clip_by_value(p_v, 1e-6, 100)

            # log GPML (marginal likelihood)
            lhood_pi_term = tf.cast(N, dtype=tf.float64) * np.log(2 * np.pi)  # ()

            mid_mat = tf.linalg.diag(varY) - tf.matmul(X, tf.matmul(A_inv, X_T))  # (1, N, N)
            Y_tilde = tf.math.reciprocal_no_nan(varY) * Y  # (1, N)
            lhood_quad_term = tf.reduce_sum(Y_tilde * tf.linalg.matvec(mid_mat, Y_tilde), axis=1)  # (1, )

            A_chol = tf.linalg.cholesky(_add_diagonal_jitter(A))  # (1, p, p)
            lhood_logdet_term = tf.reduce_sum(tf.math.log(tf.math.sqrt(varY)), axis=1) + \
                                2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(A_chol)), axis=1)  # (1, )

            gp_lhood = -0.5 * (lhood_pi_term + lhood_quad_term + lhood_logdet_term)  # (1, )

            # add C_tilde terms
            if not omit_C_tilde:
                gp_lhood = gp_lhood + tf.reduce_sum(tf.log(C_tilde + EPSILON))  # (1, )

        else:
            # Prepare all constants
            batch = tf.shape(X)[0]
            n = tf.shape(X)[1]
            ns = tf.shape(X_test)[1]

            # K_x + \sigma_x^*
            K = self.kernel_matrix_correlated_object_priors(X, X)  # (batch, n n)
            K = K + tf.matrix_diag(varY)  # (batch, n, n)
            chol_K = tf.linalg.cholesky(K)  # (batch, n, n)

            # no cholesky_solve implementation
            # inv_K = tf.linalg.inv(_add_diagonal_jitter(K, 1e-2))

            # lhood term 1/3
            lhood_pi_term = tf.cast(n, dtype=self.dtype) * np.log(2 * np.pi)

            # lhood term 2/3
            lhood_logdet_term = 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_K)), 1)  # (batch)

            # lhood term 3/3
            Y = tf.expand_dims(Y, 2)  # (batch, n, 1)
            iKY = tf.cholesky_solve(_add_diagonal_jitter(chol_K), Y)  # (batch, n, 1)
            lh_quad_term = tf.matmul(tf.transpose(Y, (0, 2, 1)), iKY)  # (batch, 1, 1)
            lh_quad_term = tf.reshape(lh_quad_term, [batch])

            # no cholesky_solve implementation
            # iKY = tf.linalg.matvec(inv_K, Y)
            # lh_quad_term = tf.matmul(iKY, tf.transpose(Y, (1, 0)))  # (batch, 1, 1)
            # lh_quad_term = tf.reshape(lh_quad_term, [batch])


            # log P(Y|X) = -1/2 * ( n log(2 pi) + Y inv(K+noise) Y + log det(K+noise))
            gp_lhood = -0.5 * (lhood_pi_term + lh_quad_term + lhood_logdet_term)

            # add C_tilde terms
            if not omit_C_tilde:
                gp_lhood = gp_lhood + tf.reduce_sum(tf.log(C_tilde + EPSILON))

            # Compute posterior mean and variances
            Ks = self.kernel_matrix_correlated_object_priors(X, X_test)  # (batch, n, ns)
            Ks_t = tf.transpose(Ks, (0, 2, 1))  # (batch, ns, n)

            # posterior mean
            p_m = tf.matmul(Ks_t, iKY)

            # no cholesky_solve implementation
            # p_m = tf.matmul(Ks_t, tf.expand_dims(iKY, 2))

            p_m = tf.reshape(p_m, (batch, ns))

            # posterior variance
            iK_Ks = tf.cholesky_solve(_add_diagonal_jitter(chol_K), Ks)  # (batch, n, ns)
            Ks_iK_Ks = tf.reduce_sum(Ks * iK_Ks, axis=1)  # (batch, ns)

            # no cholesky_solve implementation
            # Ks_iK_Ks = 1 - tf.linalg.diag_part(tf.matmul(Ks, tf.matmul(inv_K, Ks)))

            p_v = 1 - Ks_iK_Ks  # (batch, ns)
            p_v = tf.reshape(p_v, (batch, ns))
            p_v = tf.clip_by_value(p_v, 1e-6, 100)

        # drop first axis
        p_m = tf.squeeze(p_m)
        p_v = tf.squeeze(p_v)
        gp_lhood = tf.squeeze(gp_lhood)

        return p_m, p_v, gp_lhood


def forward_pass_FGPVAE_rotated_mnist(data_batch, beta, vae, GP, N_t, clipping_qs,
                                       bayes_reg_view, omit_C_tilde, C_ma, lagrange_mult, alpha,
                                       kappa, GECO=False):
    """

    :param data_batch:
    :param beta:
    :param vae:
    :param GP:
    :param N_t:
    :param clipping_qs:
    :param bayes_reg_view: whether or not to use Bayesian regresion view for linear kernel in global channels
    :param omit_C_tilde: omit C_tilde from derivation and modify cross-entropy term instead
    :param C_ma: average constraint from t-1 step (GECO)
    :param lagrange_mult: lambda from t-1 step (GECO)
    :param kappa: reconstruction level parameter for GECO
    :param alpha: moving average parameter for GECO
    :param GECO: whether or not to use GECO algorithm for training

    :return:
    """
    images, aux_data = data_batch
    aux_data = tf.reshape(aux_data, (-1, N_t, 10))

    L = vae.L
    L_w = GP.L_w
    w = tf.shape(images)[1]
    h = tf.shape(images)[2]
    K = tf.cast(w, dtype=tf.float64) * tf.cast(h, dtype=tf.float64)
    b = tf.cast(tf.shape(images)[0], dtype=tf.float64)  # batch_size

    # ENCODER NETWORK
    qnet_mu, qnet_var = vae.encode(images)
    qnet_mu = tf.reshape(qnet_mu, (-1, N_t, L))
    qnet_var = tf.reshape(qnet_var, (-1, N_t, L))

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-3, 100)

    # GP
    p_m, p_v, lhoods_local, lhoods_global = [], [], [], []

    for i in range(L_w):  # fit local GPs
        p_m_i, p_v_i, lhood_i, K_local = GP.build_1d_gp_local(X=aux_data[:, :, 1], Y=qnet_mu[:, :, i],
                                                     varY=qnet_var[:, :, i], X_test=aux_data[:, :, 1])
        p_m.append(p_m_i)
        p_v.append(p_v_i)
        lhoods_local.append(lhood_i)

    ce_global_arr = []
    for i in range(L_w, L):  # fit global GPs
        if GP.object_prior_corr:
            object_aux_data_filtered = tf.transpose(aux_data[:, ::N_t, :], perm=[1, 0, 2])
            bar_means, bar_vars, C_tilde = GP.preprocess_1d_gp_global_correlated_object_priors(qnet_mu[:, :, i],
                                                                                               qnet_var[:, :, i])
            p_m_i, p_v_i, lhood_i = GP.build_1d_gp_global_correlated_object_priors(object_aux_data_filtered,
                                                                                   bar_means,
                                                                                   bar_vars,
                                                                                   object_aux_data_filtered,
                                                                                   C_tilde,
                                                                                   bayesian_reg_view=bayes_reg_view,
                                                                                   omit_C_tilde=omit_C_tilde)

            if omit_C_tilde:
                ce_global_i = gauss_cross_entropy(p_m_i, p_v_i, bar_means, bar_vars)
                ce_global_arr.append(ce_global_i)

        else:
            p_m_i, p_v_i, lhood_i = GP.build_1d_gp_global(means=qnet_mu[:, :, i], vars=qnet_var[:, :, i])

        # repeat p_m_i and p_v_i N_t times, since those are shared across all images within one object dataset D_t
        p_m_i = tf.tile(tf.expand_dims(p_m_i, 1), [1, N_t])
        p_v_i = tf.tile(tf.expand_dims(p_v_i, 1), [1, N_t])

        p_m.append(p_m_i)
        p_v.append(p_v_i)
        lhoods_global.append(lhood_i)

    p_m = tf.stack(p_m, axis=2)
    p_v = tf.stack(p_v, axis=2)

    if GP.object_prior_corr:
        # for local channels sum over latent channels and over digits' datasets
        # for global channels we only sum over latent channels (as there is only one global GP per channel)
        lhoods = tf.reduce_sum(lhoods_local, axis=(0, 1)) + tf.reduce_sum(lhoods_global, axis=0)

        # CE (cross-entropy)
        if omit_C_tilde:
            ce_global = tf.reduce_sum(ce_global_arr)
            ce_local = gauss_cross_entropy(p_m[:, :, :L_w], p_v[:, :, :L_w], qnet_mu[:, :, :L_w], qnet_var[:, :, :L_w])
            ce_local = tf.reduce_sum(ce_local, (0, 1, 2))  # sum also over digits' datasets
            ce_term = ce_global + ce_local
        else:
            ce_term = gauss_cross_entropy(p_m, p_v, qnet_mu, qnet_var)
            ce_term = tf.reduce_sum(ce_term, (0, 1, 2))  # sum also over digits' datasets

        # KL part
        elbo_kl_part = lhoods - ce_term

    else:
        lhoods = lhoods_global + lhoods_local
        lhoods = tf.reduce_sum(lhoods, axis=0)

        # CE (cross-entropy)
        ce_term = gauss_cross_entropy(p_m, p_v, qnet_mu, qnet_var)
        ce_term = tf.reduce_sum(ce_term, (1, 2))

        # KL part
        elbo_kl_part = lhoods - ce_term

    # SAMPLE
    epsilon = tf.random.normal(shape=tf.shape(p_m), dtype=tf.float64)
    latent_samples = p_m + epsilon * tf.sqrt(p_v)

    # DECODER NETWORK (Gaussian observational likelihood, MSE)
    recon_images = vae.decode(tf.reshape(latent_samples, (-1, L)))

    if GP.object_prior_corr:
        if GECO:
            recon_loss = tf.reduce_sum((tf.reshape(images, (-1, N_t, w, h)) - tf.reshape(recon_images,
                                                                                         (-1, N_t, w, h))) ** 2,
                                       axis=[2, 3])
            recon_loss = tf.reduce_sum(recon_loss/K - kappa**2)
            C_ma = alpha * C_ma + (1 - alpha) * recon_loss / b

            # elbo = - (1/L) * KL_term + lagrange_mult * C_ma
            # elbo = - (1/b) * KL_term + lagrange_mult * C_ma
            # elbo = - KL_term + lagrange_mult * C_ma
            elbo = - elbo_kl_part + lagrange_mult * (recon_loss / b + tf.stop_gradient(C_ma - recon_loss / b))

            lagrange_mult = lagrange_mult * tf.exp(C_ma)
        else:
            recon_loss = tf.reduce_sum((tf.reshape(images, (-1, N_t, w, h)) - tf.reshape(recon_images,
                                                                                         (-1, N_t, w, h))) ** 2,
                                       axis=[1, 2, 3])
            recon_loss = tf.reduce_sum(recon_loss) / K
            elbo = -recon_loss + (beta / L) * elbo_kl_part

    else:

        if GECO:
            recon_loss = tf.reduce_mean((tf.reshape(images, (-1, N_t, w, h)) - tf.reshape(recon_images,
                                                                                          (-1, N_t, w, h))) ** 2,
                                        axis=[2, 3])
            N_t = tf.cast(N_t, dtype=tf.float64)

            C_ma = alpha * C_ma + (1 - alpha) * tf.reduce_mean(recon_loss - kappa ** 2)
            recon_loss = tf.reduce_sum(recon_loss - kappa ** 2)

            # elbo = - (1/L) * elbo_kl_part + lagrange_mult * C_ma
            # elbo = - (1/b) * elbo_kl_part + lagrange_mult * C_ma
            # elbo = - elbo_kl_part + lagrange_mult * C_ma
            elbo = - elbo_kl_part + lagrange_mult * (recon_loss / N_t + tf.stop_gradient(C_ma - recon_loss / N_t))

            lagrange_mult = lagrange_mult * tf.exp(C_ma)

        else:
            recon_loss = tf.reduce_sum((tf.reshape(images, (-1, N_t, w, h)) - tf.reshape(recon_images,
                                                                                         (-1, N_t, w, h))) ** 2,
                                       axis=[1, 2, 3])
            # ELBO
            # beta plays role of sigma_gaussian_decoder here (\lambda(\sigma_y) in Casale paper)
            # K and L are not part of ELBO. They are used in loss objective to account for the fact that magnitudes of
            # reconstruction and KL terms depend on number of pixels (K) and number of latent GPs used (L), respectively
            recon_loss = recon_loss / K
            elbo = -recon_loss + (beta/L) * elbo_kl_part

        # average across object datasets
        elbo = tf.reduce_sum(elbo)
        elbo_kl_part = tf.reduce_sum(elbo_kl_part)
        recon_loss = tf.reduce_sum(recon_loss)

    return elbo, recon_loss, elbo_kl_part, p_m, p_v, qnet_mu, qnet_var, recon_images, latent_samples, C_ma, lagrange_mult


def predict_FGPVAE_rotated_mnist(test_images, test_aux_data, train_images, train_aux_data, vae, GP,
                                  bayes_reg_view, omit_C_tilde, N_t=15, clipping_qs=False):
    """
    Get FGPVAE predictions for rotated MNIST test data.

    :param test_data_batch:
    :param train_images:
    :param train_aux_data:
    :param vae:
    :param GP:
    :param N_t:
    :param clipping_qs:
    :return:
    """

    L = vae.L
    L_w = GP.L_w
    w = tf.shape(train_images)[1]
    h = tf.shape(train_images)[2]
    train_aux_data = tf.reshape(train_aux_data, (-1, N_t, 10))
    test_aux_data = tf.expand_dims(test_aux_data, 1)

    # encode train images
    qnet_mu, qnet_var = vae.encode(train_images)
    qnet_mu = tf.reshape(qnet_mu, (-1, N_t, L))
    qnet_var = tf.reshape(qnet_var, (-1, N_t, L))

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-3, 100)

    # GP, get latent embeddings for test images
    p_m, p_v = [], []

    for i in range(L_w):  # fit local GPs
        p_m_i, p_v_i, _ , _= GP.build_1d_gp_local(X=train_aux_data[:, :, 1], Y=qnet_mu[:, :, i],
                                                     varY=qnet_var[:, :, i], X_test=test_aux_data[:, :, 1])
        p_m.append(p_m_i)
        p_v.append(p_v_i)

    for i in range(L_w, L):  # fit global GPs
        if GP.object_prior_corr:
            object_aux_data_filtered = tf.transpose(train_aux_data[:, ::N_t, :], perm=[1, 0, 2])
            bar_means, bar_vars, C_tilde = GP.preprocess_1d_gp_global_correlated_object_priors(qnet_mu[:, :, i],
                                                                                               qnet_var[:, :, i])
            p_m_i, p_v_i, _ = GP.build_1d_gp_global_correlated_object_priors(object_aux_data_filtered,
                                                                             bar_means,
                                                                             bar_vars,
                                                                             object_aux_data_filtered,
                                                                             C_tilde,
                                                                             omit_C_tilde=omit_C_tilde,
                                                                             bayesian_reg_view=bayes_reg_view)
        else:
            p_m_i, p_v_i, _ = GP.build_1d_gp_global(means=qnet_mu[:, :, i], vars=qnet_var[:, :, i])

        p_m.append(tf.expand_dims(p_m_i, 1))
        p_v.append(tf.expand_dims(p_v_i, 1))

    p_m = tf.stack(p_m, axis=2)
    p_v = tf.stack(p_v, axis=2)

    # SAMPLE
    epsilon = tf.random.normal(shape=tf.shape(p_m), dtype=tf.float64)
    latent_samples = p_m + epsilon * tf.sqrt(p_v)

    # decode, calculate error (Gaussian observational likelihood, MSE)
    recon_images = vae.decode(tf.reshape(latent_samples, (-1, L)))
    recon_loss = tf.reduce_mean((test_images - recon_images) ** 2)

    return recon_images, recon_loss


def extrapolate_experiment_eval_data(mnist_path, digit, N_t, pred_angle_id=7, nr_angles=16):
    """
    Prepare validation dataset for the extrapolate experiment.


    :param mnist_path:
    :param digit:
    :param N_t: how many angles do we observe for each image in test set
    :param pred_angle_id: which angle to leave out for prediction
    :param nr_angles: size of object dataset
    :return:
    """

    eval_data_dict = pickle.load(open(mnist_path + 'eval_data{}_not_shuffled.p'.format(digit), 'rb'))

    eval_images, eval_aux_data = eval_data_dict["images"], eval_data_dict["aux_data"]
    pred_angle_mask = [pred_angle_id + i * nr_angles for i in range(int(len(eval_aux_data) / nr_angles))]
    not_pred_angle_mask = [i for i in range(len(eval_images)) if i not in pred_angle_mask]

    observed_images = eval_images[not_pred_angle_mask]
    observed_aux_data = eval_aux_data[not_pred_angle_mask]

    # randomly drop some observed angles
    if N_t < 15:
        digit_mask = [True]*N_t + [False]*(15-N_t)
        mask = [random.sample(digit_mask, len(digit_mask)) for _ in range(int(len(eval_aux_data)/nr_angles))]
        flatten = lambda l: [item for sublist in l for item in sublist]
        mask = flatten(mask)

        observed_images = observed_images[mask]
        observed_aux_data = observed_aux_data[mask]

    test_images = eval_images[pred_angle_mask]
    test_aux_data = eval_aux_data[pred_angle_mask]

    return observed_images, observed_aux_data, test_images, test_aux_data


def latent_samples_FGPVAE(train_images, train_aux_data, vae, GP, N_t, clipping_qs=False):
    """
    Get latent samples for training data. For t-SNE plots :)

    :param train_images:
    :param train_aux_data:
    :param vae:
    :param GP:
    :param clipping_qs:
    :return:
    """

    train_aux_data = tf.reshape(train_aux_data, (-1, N_t, 10))
    L = vae.L
    L_w = GP.L_w

    # ENCODER NETWORK
    qnet_mu, qnet_var = vae.encode(train_images)
    qnet_mu = tf.reshape(qnet_mu, (-1, N_t, L))
    qnet_var = tf.reshape(qnet_var, (-1, N_t, L))

    # clipping of VAE posterior variance
    if clipping_qs:
        qnet_var = tf.clip_by_value(qnet_var, 1e-3, 100)

    # GP
    p_m, p_v = [], []

    for i in range(L_w):  # fit local GPs
        p_m_i, p_v_i, _, _ = GP.build_1d_gp_local(X=train_aux_data[:, :, 1], Y=qnet_mu[:, :, i],
                                                  varY=qnet_var[:, :, i], X_test=train_aux_data[:, :, 1])
        p_m.append(p_m_i)
        p_v.append(p_v_i)

    for i in range(L_w, L):  # fit global GPs
        p_m_i, p_v_i, lhood_i = GP.build_1d_gp_global(means=qnet_mu[:, :, i], vars=qnet_var[:, :, i])

        # repeat p_m_i and p_v_i N_t times, since those are shared across all images within one object dataset D_t
        p_m_i = tf.tile(tf.expand_dims(p_m_i, 1), [1, N_t])
        p_v_i = tf.tile(tf.expand_dims(p_v_i, 1), [1, N_t])

        p_m.append(p_m_i)
        p_v.append(p_v_i)

    p_m = tf.stack(p_m, axis=2)
    p_v = tf.stack(p_v, axis=2)

    # SAMPLE
    epsilon = tf.random.normal(shape=tf.shape(p_m), dtype=tf.float64)
    latent_samples = p_m + epsilon * tf.sqrt(p_v)

    return latent_samples

