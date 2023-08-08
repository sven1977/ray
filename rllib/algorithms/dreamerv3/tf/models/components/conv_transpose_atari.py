"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""
from typing import Optional

import numpy as np

from ray.rllib.algorithms.dreamerv3.utils import get_cnn_multiplier
from ray.rllib.utils.framework import try_import_tf

_, tf, _ = try_import_tf()


class ConvTransposeAtari(tf.keras.Model):
    """A Conv2DTranspose decoder to generate Atari images from a latent space.

    Wraps an initial single linear layer with a stack of 4 Conv2DTranspose layers (with
    layer normalization) and a diag Gaussian, from which we then sample the final image.
    Sampling is done with a fixed stddev=1.0 and using the mean values coming from the
    last Conv2DTranspose layer.
    """

    def __init__(
        self,
        *,
        input_size: int,
        model_size: Optional[str] = "XS",
        cnn_multiplier: Optional[int] = None,
        gray_scaled: bool,
    ):
        """Initializes a ConvTransposeAtari instance.

        Args:
            input_size: The size (int) of the input tensor.
            model_size: The "Model Size" used according to [1] Appendinx B.
                Use None for manually setting the `cnn_multiplier`.
            cnn_multiplier: Optional override for the additional factor used to multiply
                the number of filters with each CNN transpose layer. Starting with
                8 * `cnn_multiplier` filters in the first CNN transpose layer, the
                number of filters then decreases via `4*cnn_multiplier`,
                `2*cnn_multiplier`, till `1*cnn_multiplier`.
            gray_scaled: Whether the last Conv2DTranspose layer's output has only 1
                color channel (gray_scaled=True) or 3 RGB channels (gray_scaled=False).
        """
        super().__init__(name="image_decoder")

        cnn_multiplier = get_cnn_multiplier(model_size, override=cnn_multiplier)

        # The shape going into the first Conv2DTranspose layer.
        # We start with a 4x4 channels=8 "image".
        self.input_dims = (4, 4, 8 * cnn_multiplier)

        self.gray_scaled = gray_scaled

        # See appendix B in [1]:
        # "The decoder starts with a dense layer, followed by reshaping
        # to 4 × 4 × C and then inverts the encoder architecture. ..."
        layers = [
            tf.keras.layers.Input((input_size,)),

            tf.keras.layers.Dense(
                units=int(np.prod(self.input_dims)),
                activation=None,
                use_bias=True,
            ),

            # Reshape to image format.
            tf.keras.layers.Reshape(self.input_dims),

            # Inverse conv2d stack. See cnn_atari.py for corresponding Conv2D stack.
            # Create one LayerNorm layer for each of the Conv2DTranspose layers.
            tf.keras.layers.Conv2DTranspose(
                filters=4 * cnn_multiplier,
                kernel_size=4,
                strides=(2, 2),
                padding="same",
                # No bias or activation due to layernorm.
                activation=None,
                use_bias=False,
            ),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("silu"),

            tf.keras.layers.Conv2DTranspose(
                filters=2 * cnn_multiplier,
                kernel_size=4,
                strides=(2, 2),
                padding="same",
                # No bias or activation due to layernorm.
                activation=None,
                use_bias=False,
            ),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("silu"),

            tf.keras.layers.Conv2DTranspose(
                filters=1 * cnn_multiplier,
                kernel_size=4,
                strides=(2, 2),
                padding="same",
                # No bias or activation due to layernorm.
                activation=None,
                use_bias=False,
            ),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Activation("silu"),

            # Important! No activation or layer norm for last layer as the outputs of
            # this one go directly into the diag-gaussian as parameters.
            tf.keras.layers.Conv2DTranspose(
                filters=1 if self.gray_scaled else 3,
                kernel_size=4,
                strides=(2, 2),
                padding="same",
                activation=None,
                use_bias=True,  # Last layer does use bias (b/c has no LayerNorm).
            )
            # .. until output is 64 x 64 x 3 (or 1 for self.gray_scaled=True).
        ]

        self.net = tf.keras.models.Sequential(layers)

    def call(self, h, z):
        """Performs a forward pass through the Conv2D transpose decoder.

        Args:
            h: The deterministic hidden state of the sequence model.
            z: The sequence of stochastic discrete representations of the original
                observation input. Note: `z` is not used for the dynamics predictor
                model (which predicts z from h).
        """
        # Flatten last two dims of z.
        assert len(z.shape) == 3
        z_shape = tf.shape(z)
        z = tf.reshape(tf.cast(z, tf.float32), shape=(z_shape[0], -1))
        assert len(z.shape) == 2
        input_ = tf.concat([h, z], axis=-1)

        # Pass through stack of Conv2DTransport layers (and layer norms).
        out = self.net(input_)
        out += 0.5  # See Danijar's code
        out_shape = tf.shape(out)

        # Interpret output as means of a diag-Gaussian with std=1.0:
        # From [2]:
        # "Distributions: The image predictor outputs the mean of a diagonal Gaussian
        # likelihood with unit variance, ..."

        # Reshape `out` for the diagonal multi-variate Gaussian (each pixel is its own
        # independent (b/c diagonal co-variance matrix) variable).
        loc = tf.reshape(out, shape=(out_shape[0], -1))

        return loc
