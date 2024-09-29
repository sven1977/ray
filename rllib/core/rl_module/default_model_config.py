from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union


@dataclass
class DefaultModelConfig:
    """
    Attributes:
        encoder_fcnet_hiddens: List indicating the number of Linear layers and their
            sizes in the encoder component.
    """
    encoder_fcnet_hiddens: List[int] = field(default_factory=lambda: [256, 256])
    # Activation function descriptor.
    # Supported values are: "tanh", "relu", "swish" (or "silu", which is the same),
    # "linear" (or None).
    encoder_fcnet_activation: str = "relu"
    # Initializer function or class descriptor for encoder weigths.
    # Supported values are the initializer names (str), classes or functions listed
    # by the frameworks (`tf2`, `torch`). See
    # https://pytorch.org/docs/stable/nn.init.html for `torch` and
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers for `tf2`.
    # Note, if `None`, the default initializer defined by `torch` or `tf2` is used.
    encoder_fcnet_weights_initializer: Optional[Union[str, Callable]] = None
    # Initializer configuration for encoder weights.
    # This configuration is passed to the initializer defined in
    # `fcnet_weights_initializer`.
    encoder_fcnet_weights_initializer_config: Optional[dict] = None
    # Initializer function or class descriptor for encoder bias.
    # Supported values are the initializer names (str), classes or functions listed
    # by the frameworks (`tf2``, `torch`). See
    # https://pytorch.org/docs/stable/nn.init.html for `torch` and
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers for `tf2`.
    # Note, if `None`, the default initializer defined by `torch` or `tf2` is used.
    encoder_fcnet_bias_initializer: Optional[Union[str, Callable]] = None
    # Initializer configuration for encoder bias.
    # This configuration is passed to the initializer defined in
    # `fcnet_bias_initializer`.
    encoder_fcnet_bias_initializer_config: Optional[dict] = None
