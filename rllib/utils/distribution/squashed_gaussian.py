import numpy as np

from ray.rllib.utils.annotations import override
from ray.rllib.utils.distribution.distribution import Distribution
from ray.rllib.utils.framework import try_import_tf, try_import_tfp, \
    try_import_torch
from ray.rllib.utils.numpy import MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT, \
    SMALL_NUMBER

tf = try_import_tf()
tfp = try_import_tfp()
torch, _ = try_import_torch()


class SquashedGaussian(Distribution):
    """A tanh-squashed Gaussian distribution defined by: mean, std, low, high.

    The distribution will never return low or high exactly, but
    `low`+SMALL_NUMBER or `high`-SMALL_NUMBER respectively.
    """
    def __init__(self, inputs, model, low=-1.0, high=1.0, framework="tf"):
        """Parameterizes the distribution via `inputs`.

        Args:
            low (float): The lowest possible sampling value
                (excluding this value).
            high (float): The highest possible sampling value
                (excluding this value).
        """
        assert tfp is not None
        assert np.all(np.less(low, high))
        self.low = low
        self.high = high
        super().__init__(inputs, model, framework=framework)

        if framework == "tf":
            self.sampled_action_logp = self._sampled_action_logp_tf
            self.logp = self._logp_tf
        else:
            self.sampled_action_logp = self._sampled_action_logp_torch
            self.logp = self._logp_torch

    @override(Distribution)
    def deterministic_sample(self):
        mean = self.distr.mean()
        return self._squash(mean)

    @override(Distribution)
    def required_model_output_shape(self, space, model_config):
        return np.prod(space.shape) * 2

    def _sampled_action_logp_tf(self):
        unsquashed_values = self._unsquash_tf(self.sample_op)
        log_prob = tf.reduce_sum(
            self.distr.log_prob(unsquashed_values), axis=-1)
        unsquashed_values_tanhd = tf.math.tanh(unsquashed_values)
        log_prob -= tf.math.reduce_sum(
            tf.math.log(1 - unsquashed_values_tanhd**2 + SMALL_NUMBER),
            axis=-1)
        return log_prob

    def _sampled_action_logp_torch(self):
        unsquashed_values = self._unsquash_torch(self.sample_op)
        log_prob = torch.sum(
            self.distr.log_prob(unsquashed_values), axis=-1)
        unsquashed_values_tanhd = torch.tanh(unsquashed_values)
        log_prob -= torch.sum(
            torch.log(1 - unsquashed_values_tanhd**2 + SMALL_NUMBER),
            axis=-1)
        return log_prob

    def _logp_tf(self, x):
        unsquashed_values = self._unsquash_tf(x)
        log_prob = tf.reduce_sum(
            self.distr.log_prob(value=unsquashed_values), axis=-1)
        unsquashed_values_tanhd = tf.math.tanh(unsquashed_values)
        log_prob -= tf.math.reduce_sum(
            tf.math.log(1 - unsquashed_values_tanhd**2 + SMALL_NUMBER),
            axis=-1)
        return log_prob

    def _logp_torch(self, x):
        unsquashed_values = self._unsquash_torch(x)
        log_prob = torch.sum(
            self.distr.log_prob(value=unsquashed_values), axis=-1)
        unsquashed_values_tanhd = torch.tanh(unsquashed_values)
        log_prob -= torch.sum(
            torch.log(1 - unsquashed_values_tanhd**2 + SMALL_NUMBER),
            axis=-1)
        return log_prob

    @override(Distribution)
    def _build_tf_sample_op(self):
        return self._squash_tf(self.distr.sample())

    @override(Distribution)
    def _get_tfp_dist(self):
        loc, log_scale = tf.split(self.inputs, 2, axis=-1)
        # Clip `scale` values (coming from NN) to reasonable values.
        log_scale = tf.clip_by_value(log_scale, MIN_LOG_NN_OUTPUT,
                                     MAX_LOG_NN_OUTPUT)
        scale = tf.exp(log_scale)
        self.distr = tfp.distributions.Normal(loc=loc, scale=scale)

    @override(Distribution)
    def _get_torch_dist(self):
        loc, log_scale = torch.split(
            self.inputs, int(self.inputs.size()[-1]/2), dim=-1)
        # Clip `scale` values (coming from NN) to reasonable values.
        log_scale = torch.clamp(log_scale, MIN_LOG_NN_OUTPUT,
                                        MAX_LOG_NN_OUTPUT)
        scale = torch.exp(log_scale)
        self.distr = torch.distributions.Normal(loc=loc, scale=scale)

    def _squash_tf(self, raw_values):
        # Make sure raw_values are not too high/low (such that tanh would
        # return exactly 1.0/-1.0, which would lead to +/-inf log-probs).
        return (tf.clip_by_value(
            tf.math.tanh(raw_values),
            -1.0 + SMALL_NUMBER,
            1.0 - SMALL_NUMBER) + 1.0) / 2.0 * (self.high - self.low) + \
               self.low

    def _squash_torch(self, raw_values):
        # Make sure raw_values are not too high/low (such that tanh would
        # return exactly 1.0/-1.0, which would lead to +/-inf log-probs).
        return (torch.clamp(
            torch.tanh(raw_values),
            -1.0 + SMALL_NUMBER,
            1.0 - SMALL_NUMBER) + 1.0) / 2.0 * (self.high - self.low) + \
               self.low

    def _unsquash_tf(self, values):
        return tf.math.atanh((values - self.low) /
                             (self.high - self.low) * 2.0 - 1.0)

    def _unsquash_torch(self, values):
        return torch.atanh((values - self.low) /
                           (self.high - self.low) * 2.0 - 1.0)
