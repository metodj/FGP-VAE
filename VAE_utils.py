import tensorflow as tf


class mnistVAE:

    dtype = tf.float64

    def __init__(self, im_width=28, im_height=28, L=16):
        """
        VAE for rotated MNIST data. Architecture (almost) same as in Casale (see Figure 4 in Supplementary material).

        :param im_width:
        :param im_height:
        :param L:
        """

        self.L = L

        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(im_width, im_height, 1), dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, strides=(2, 2), activation='elu', dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, strides=(2, 2), activation='elu', dtype=self.dtype),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, strides=(2, 2), activation='elu', dtype=self.dtype),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(2 * self.L)
            ])

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, dtype=self.dtype),
                tf.keras.layers.Reshape(target_shape=(4, 4, 8)),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=8, kernel_size=3, strides=(1, 1), activation='elu', dtype=self.dtype),
                tf.keras.layers.UpSampling2D(size=(2, 2)),
                tf.keras.layers.Conv2D(
                    filters=1, kernel_size=3, strides=(1, 1), activation='elu', padding='same', dtype=self.dtype),
            ])

    def encode(self, images):
        """

        :param images:
        :return:
        """

        encodings = self.encoder(images)
        means, vars = encodings[:, :self.L], tf.exp(encodings[:, self.L:])  # encoder outputs \mu and log(\sigma^2)
        return means, vars

    def decode(self, latent_samples):
        """

        :param latent_samples:
        :return:
        """

        recon_images = self.decoder(latent_samples)
        return recon_images



