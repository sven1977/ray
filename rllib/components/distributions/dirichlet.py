import numpy as np

from ray.rllib.components.distributions.distribution import Distribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_tfp, \
    try_import_torch

tfp = try_import_tfp()
tf = try_import_tf()
torch, _ = try_import_torch()


class Dirichlet(Distribution):
    """
    Dirichlet distribution for Simplex Spaces (values lie on an n-D simplex;
    between [0,1] and sum to 1).
    E.g.: Used for actions that represent resource allocation.
    """
    @override(Distribution)
    def __init__(self, inputs, framework="tf"):
        """
        Input is a tensor of logits. The exponential of logits is used to
        parametrize the Dirichlet distribution as all parameters need to be
        positive. An arbitrary small epsilon is added to the concentration
        parameters to be zero due to numerical error.

        See issue #4440 for more details.
        """
        self.epsilon = 1e-7
        concentration = tf.exp(inputs) + self.epsilon
        self.dist = tfp.distributions.Dirichlet(
            concentration=concentration,
            validate_args=True,
            allow_nan_stats=False,
        ) if framework == "tf" else None

        super(Dirichlet, self).__init__(self, concentration)

    @staticmethod
    @override(Distribution)
    def required_model_output_shape(action_space, model_config):
        return np.prod(action_space.shape)

    @override(Distribution)
    def _tf_logp(self, x):
        # Support of Dirichlet are positive real numbers. x is already be
        # an array of positive number, but we clip to avoid zeros due to
        # numerical errors.
        x = tf.maximum(x, self.epsilon)
        x = x / tf.reduce_sum(x, axis=-1, keepdims=True)
        return self.tf_dist.log_prob(x)

    @override(Distribution)
    def _tf_entropy(self):
        return self.tf_dist.entropy()

    @override(Distribution)
    def _tf_kl(self, other):
        assert isinstance(other.tf_dist, tfp.distributions.Dirichlet)
        return self.tf_dist.kl_divergence(other.tf_dist)

    @override(Distribution)
    def _build_tf_sample_op(self):
        return self.tf_dist.sample()

    @override(Distribution)
    def _get_torch_dist(self):
        # `self.inputs` are the concentrations for the Dirichlet.
        return torch.distributions.dirichlet.Dirichlet(self.inputs)
