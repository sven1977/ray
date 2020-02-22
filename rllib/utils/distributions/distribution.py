from abc import ABCMeta, abstractmethod

from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf = try_import_tf()
torch, _ = try_import_torch()


@DeveloperAPI
class Distribution(metaclass=ABCMeta):
    """A generic Distribution class.

    NOTE: Formerly: `ActionDistribution`.
    """
    @DeveloperAPI
    def __init__(self, inputs, model, framework="tf"):
        """
        Initializes the distribution.

        Args:
            inputs (Tensors): NN outputs to parameterize the distribution with.
            model (Model): The model using this distribution. Only used for
                auto-regressive distributions.
            framework (str): One of "tf" or "torch".
        """
        self.inputs = inputs
        self.model = model
        self.framework = framework
        self.tf_sample_op = None
        self.torch_dist = None
        self.torch_last_sample = None

        if self.framework == "tf":
            self.tf_sample_op = self._build_tf_sample_op()
        elif self.framework == "torch":
            self.inputs = torch.Tensor(self.inputs)
            self.torch_dist = self._get_torch_dist()

    @DeveloperAPI
    def sample(self):
        """
        Return a drawn a sample from the distribution given by `self.inputs`.
        """
        if self.framework == "tf":
            if tf.executing_eagerly():
                return self._build_tf_sample_op()
            return self.tf_sample_op
        else:
            # Store the last sample to serve calls to `self.sample_logp`.
            self.torch_last_sample = self.torch_dist.sample()
            return self.torch_last_sample

    @abstractmethod
    @DeveloperAPI
    def deterministic_sample(self):
        """
        Get the deterministic "sampling" output from the distribution.
        This is usually the max likelihood output, i.e. mean for Normal,
        argmax for Categorical, etc..
        """
        raise NotImplementedError

    @DeveloperAPI
    def sampled_logp(self):
        """
        Returns:
             any: The log probability of the last sample.
        """
        if self.framework == "tf":
            if tf.executing_eagerly():
                return self.logp(self._build_tf_sample_op())
            return self.logp(self.sample_op)
        else:
            assert self.torch_last_sample is not None
            return self.logp(self.torch_last_sample)

    @DeveloperAPI
    def logp(self, x):
        """
        Returns:
            The log-likelihood of x given the parameterized Distribution.
        """
        if self.framework == "tf":
            return self._tf_logp(x)
        else:
            return self.torch_dist.log_prob(torch.Tensor(x))

    @DeveloperAPI
    def kl(self, other):
        """
        Returns:
            The KL-divergence between two Distributions.
        """
        if self.framework == "tf":
            return self._tf_kl(other)
        else:
            return torch.distributions.kl.kl_divergence(
                self.torch_dist, other.torch_dist
            )

    @DeveloperAPI
    def entropy(self):
        """
        The entropy of the Distribution.
        """
        if self.framework == "tf":
            return self._tf_entropy()
        else:
            return self.torch_dist.entropy()

    def multi_kl(self, other):
        """
        The KL-divergence between two Distributions.
        This differs from kl() in that it can return an array for
        MultiDiscrete. TODO(ekl) consider removing this.
        """
        return self.kl(other)

    def multi_entropy(self):
        """
        The entropy of the Distribution.
        This differs from entropy() in that it can return an array for
        MultiDiscrete. TODO(ekl) consider removing this.
        """
        return self.entropy()

    @staticmethod
    @abstractmethod
    @DeveloperAPI
    def required_model_output_shape(space, model_config):
        """
        Returns the required shape of an input parameter tensor for a
        particular Space and an optional dict of distribution-specific
        options.

        Args:
            space (gym.Space): The space this distribution will
                be used for, whose shape attributes will be used to determine
                the required shape of the input parameter tensor.
            model_config (dict): Model's config dict (as defined in catalog.py)

        Returns:
            Union[int,np.ndarray[int]]: Shape of the required input vector
                (w/o the leading batch dimension).
        """
        raise NotImplementedError

    @abstractmethod
    def _build_tf_sample_op(self):
        """
        Returns:
            tfp.??: The tfp.distributions sampling op given `self.inputs`.
        """
        raise NotImplementedError

    @abstractmethod
    def _tf_logp(self, x):
        """
        Args:
            x (tf.Tensor): The sample(s) to get the logp value(s) for.

        Returns:
            tf.Tensor: The log-likelihood of the sampled `x`, given
                `self.inputs`.
        """
        raise NotImplementedError

    @abstractmethod
    def _tf_entropy(self):
        """
        Returns:
            tf.Tensor: The entropy of the distribution given `self.inputs`.
        """
        raise NotImplementedError

    @abstractmethod
    def _tf_kl(self, other):
        """
        Args:
            other (Distribution): The other distribution to calculate the
                KL-divergence against.

        Returns:
            tf.Tensor: The KL-divergence between this dist and `other`.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_torch_dist(self):
        """
        Returns:
            torch.distributions.Distribution: The torch dist object to use
                given `self.inputs`.
        """
        raise NotImplementedError
