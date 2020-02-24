from gym.spaces import Discrete, MultiDiscrete

from ray.rllib.utils.annotations import override
from ray.rllib.utils.distribution.distribution import Distribution
from ray.rllib.utils.framework import try_import_tf, try_import_tfp, \
    try_import_torch

tf = try_import_tf()
tfp = try_import_tfp()
torch, _ = try_import_torch()


class Bernoulli(Distribution):
    """A binary distribution defined by the probability for the value=True.
    
    IMPORTANT NOTE:
    Bernoulli.entropy calculates the Shannon entropy (- SUM(i) pi * log(pi)),
    but with the natural log (ln), rather than log2!
    This is documented incorrectly in the tfp documentation.
    """
    @override(Distribution)
    def __init__(self, inputs, model, framework="tf"):
        assert tfp, "TFP must be installed to use a Bernoulli distribution!"
        super().__init__(inputs, model, framework=framework)

    @override(Distribution)
    def deterministic_sample(self):
        return self.dist.probs >= 0.5

    @override(Distribution)
    def required_model_output_shape(self, space, model_config):
        assert isinstance(space, (Discrete, MultiDiscrete))
        return 1

    @override(Distribution)
    def _get_tfp_dist(self):
        return tfp.distributions.Bernoulli(logits=self.inputs, dtype=tf.bool)

    @override(Distribution)
    def _get_torch_dist(self):
        return torch.distributions.Bernoulli(logits=self.inputs)
