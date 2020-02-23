import numpy as np

from ray.rllib.utils.distribution.categorical import Categorical
from ray.rllib.utils.distribution.distribution import Distribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf, try_import_torch

tf = try_import_tf()
torch, nn = try_import_torch()


class MultiCategorical(Distribution):
    """
    Categorical distribution parameterized by logits (softmaxed to n discrete
    probabilities summing to 1.0).
    """
    @override(Distribution)
    def __init__(self, inputs, model, framework="tf"):
        self.cats = [
            Categorical(input_, model, framework=framework)
            for input_ in tf.split(inputs, input_lens, axis=1)
        ] if framework == "tf" else None
        super().__init__(inputs, model, framework=framework)

    @override(Distribution)
    def deterministic_sample(self):
        if self.framework == "tf":
            return tf.math.argmax(self.inputs, axis=-1)
        else:
            return self.dist.probs.argmax(dim=-1)

    @override(Distribution)
    def logp(self, x):
        logp = super().logp(x)
        if self.framework == "tf":
            return logp
        else:
            return logp.sum(-1)

    @override(Distribution)
    def multi_entropy(self):
        if self.framework == "tf":
            return tf.stack([cat.entropy() for cat in self.cats], axis=1)
        else:
            # TODO:
            return tf.stack([cat.entropy() for cat in self.cats], axis=1)

    @staticmethod
    @override(Distribution)
    def required_model_output_shape(space, model_config):
        return np.sum(space.nvec)

    @override(Distribution)
    def _build_tf_sample_op(self):
        return tf.stack([cat.sample() for cat in self.cats], axis=1)

    @override(Distribution)
    def _tf_logp(self, x):
        # If tensor is provided, unstack it into list.
        if isinstance(x, tf.Tensor):
            x = tf.unstack(tf.cast(x, tf.int32), axis=1)
        logps = tf.stack([cat.logp(act) for cat, act in zip(self.cats, x)])
        return tf.reduce_sum(logps, axis=0)

    @override(Distribution)
    def _tf_entropy(self):
        return tf.reduce_sum(self.multi_entropy(), axis=1)

    @override(Distribution)
    def _tf_kl(self, other):
        return tf.reduce_sum(self.multi_kl(other), axis=1)

    @override(Distribution)
    def _get_torch_dist(self):
        return torch.distributions.categorical.Categorical(logits=self.inputs)
