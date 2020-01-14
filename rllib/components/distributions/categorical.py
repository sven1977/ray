from ray.rllib.components.distributions.distribution import Distribution
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_tf, try_import_torch

tf = try_import_tf()
torch, nn = try_import_torch()


class Categorical(Distribution):
    """
    Categorical distribution parameterized by logits (softmaxed to n discrete
    probabilities summing to 1.0).
    """
    @override(Distribution)
    def deterministic_sample(self):
        if self.framework == "tf":
            return tf.math.argmax(self.inputs, axis=1)
        else:
            return self.dist.probs.argmax(dim=1)

    @override(Distribution)
    def logp(self, x):
        logp = super().logp(x)
        if self.framework == "tf":
            return logp
        else:
            return logp.sum(-1)

    @staticmethod
    @override(Distribution)
    def required_model_output_shape(space, model_config):
        return space.n

    @override(Distribution)
    def _build_tf_sample_op(self):
        # TODO: use tfp.
        return tf.squeeze(tf.multinomial(self.inputs, 1), axis=1)

    @override(Distribution)
    def _tf_logp(self, x):
        # TODO: use tfp
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.inputs, labels=tf.cast(x, tf.int32))

    @override(Distribution)
    def _tf_entropy(self):
        # TODO: use tfp
        a0 = self.inputs - tf.reduce_max(
            self.inputs, axis=[1], keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=[1], keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis=[1])

    @override(Distribution)
    def _tf_kl(self, other):
        # TODO: use tfp.
        a0 = self.inputs - tf.reduce_max(
            self.inputs, axis=[1], keep_dims=True)
        a1 = other.inputs - tf.reduce_max(
            other.inputs, axis=[1], keep_dims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=[1], keep_dims=True)
        z1 = tf.reduce_sum(ea1, axis=[1], keep_dims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(
            p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=[1])

    @override(Distribution)
    def _get_torch_dist(self):
        return torch.distributions.categorical.Categorical(logits=self.inputs)
