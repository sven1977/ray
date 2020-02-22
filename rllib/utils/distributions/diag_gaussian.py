import numpy as np

from ray.rllib.utils.distributions.distribution import Distribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf, try_import_torch

tf = try_import_tf()
torch, _ = try_import_torch()


class DiagGaussian(Distribution):
    """
    Diagonal Gaussian distribution.
    Parameterized by splitting the NN output in half and interpreting the
    first half as the (diagonal) mean values and the second half as the
    (diagonal) log-stddev values.

    TODO: Should we clip the log-std output by something like [-20, 2]?
    """
    @override(Distribution)
    def __init__(self, inputs, framework="tf"):
        super(DiagGaussian, self).__init__(inputs, framework)
        if self.framework == "tf":
            self.mean, self.log_std = tf.split(inputs, 2, axis=1)
            self.std = tf.exp(self.log_std)
        else:
            self.mean, self.log_std = torch.chunk(inputs, 2, dim=1)
            self.std = None  # not needed for torch

    @override(Distribution)
    def deterministic_sample(self):
        if self.framework == "tf":
            return self.mean
        else:
            return self.dist.mean

    #@override(DistributionWrapper)
    #def logp(self, x):
    #    return TorchDistributionWrapper.logp(self, actions).sum(-1)

    @staticmethod
    @override(Distribution)
    def required_model_output_shape(space, model_config):
        return np.prod(space.shape) * 2

    @override(Distribution)
    def _build_tf_sample_op(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    @override(Distribution)
    def _tf_logp(self, x):
        return -0.5 * tf.reduce_sum(
            tf.square((x - self.mean) / self.std), axis=1
        ) - 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[1]) - \
               tf.reduce_sum(self.log_std, axis=1)

    @override(Distribution)
    def _tf_entropy(self):
        return tf.reduce_sum(
            self.log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=1
        )

    @override(Distribution)
    def _tf_kl(self, other):
        assert isinstance(other, DiagGaussian)
        return tf.reduce_sum(
            other.log_std - self.log_std +
            (tf.square(self.std) + tf.square(self.mean - other.mean)) /
            (2.0 * tf.square(other.std)) - 0.5,
            axis=1
        )

    @override(Distribution)
    def _get_torch_dist(self):
        return torch.distributions.normal.Normal(
            self.mean, torch.exp(self.log_std)
        )
