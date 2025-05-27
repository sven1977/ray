from typing import Callable, Dict, List, Optional, Union, Tuple

from ray.rllib.models.torch.misc import same_padding, valid_padding
from ray.rllib.models.utils import get_activation_fn, get_initializer_fn
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class TorchCNN(nn.Module):
    """A model containing a CNN with N Conv2D layers.

    All layers share the same activation function, bias setup (use bias or not),
    and LayerNorm setup (use layer normalization or not).

    Note that there is no flattening nor an additional dense layer at the end of the
    stack. The output of the network is a 3D tensor of dimensions
    [width x height x num output filters].
    """

    def __init__(
        self,
        *,
        input_dims: Union[List[int], Tuple[int]],
        cnn_filter_specifiers: List[List[Union[int, List]]],
        cnn_use_bias: bool = True,
        cnn_use_layernorm: bool = False,
        cnn_activation: str = "relu",
        cnn_kernel_initializer: Optional[Union[str, Callable]] = None,
        cnn_kernel_initializer_config: Optional[Dict] = None,
        cnn_bias_initializer: Optional[Union[str, Callable]] = None,
        cnn_bias_initializer_config: Optional[Dict] = None,
    ):
        """Initializes a TorchCNN instance.

        Args:
            input_dims: The 3D input dimensions of the network (incoming image).
            cnn_filter_specifiers: A list in which each element is another (inner) list
                of either the following forms:
                `[number of channels/filters, kernel, stride]`
                OR:
                `[number of channels/filters, kernel, stride, padding]`, where `padding`
                can either be "same" or "valid".
                When using the first format w/o the `padding` specifier, `padding` is
                "same" by default. Also, `kernel` and `stride` may be provided either as
                single ints (square) or as a tuple/list of two ints (width- and height
                dimensions) for non-squared kernel/stride shapes.
                A good rule of thumb for constructing CNN stacks is:
                When using padding="same", the input "image" will be reduced in size by
                the factor `stride`, e.g. input=(84, 84, 3) stride=2 kernel=x
                padding="same" filters=16 -> output=(42, 42, 16).
                For example, if you would like to reduce an Atari image from its
                original (84, 84, 3) dimensions down to (6, 6, F), you can construct the
                following stack and reduce the w x h dimension of the image by 2 in each
                layer:
                [[16, 4, 2], [32, 4, 2], [64, 4, 2], [128, 4, 2]] -> output=(6, 6, 128)
            cnn_use_bias: Whether to use bias on all Conv2D layers.
            cnn_activation: The activation function to use after each Conv2D layer.
            cnn_use_layernorm: Whether to insert a LayerNormalization functionality
                in between each Conv2D layer's outputs and its activation.
            cnn_kernel_initializer: The initializer function or class to use for kernel
                initialization in the CNN layers. If `None` the default initializer of
                the respective CNN layer is used. Note, only the in-place
                initializers, i.e. ending with an underscore "_" are allowed.
            cnn_kernel_initializer_config: Configuration to pass into the initializer
                defined in `cnn_kernel_initializer`.
            cnn_bias_initializer: The initializer function or class to use for bias
                initializationcin the CNN layers. If `None` the default initializer of
                the respective CNN layer is used. Note, only the in-place initializers,
                i.e. ending with an underscore "_" are allowed.
            cnn_bias_initializer_config: Configuration to pass into the initializer
                defined in `cnn_bias_initializer`.
        """
        super().__init__()

        assert len(input_dims) == 3

        cnn_activation = get_activation_fn(cnn_activation, framework="torch")
        cnn_kernel_initializer = get_initializer_fn(
            cnn_kernel_initializer, framework="torch"
        )
        cnn_bias_initializer = get_initializer_fn(
            cnn_bias_initializer, framework="torch"
        )
        layers = []

        # Add user-specified hidden convolutional layers first
        width, height, in_depth = input_dims
        in_size = [width, height]
        for filter_specs in cnn_filter_specifiers:
            # Padding information not provided -> Use "same" as default.
            if len(filter_specs) == 3:
                out_depth, kernel_size, strides = filter_specs
                padding = "same"
            # Padding information provided.
            else:
                out_depth, kernel_size, strides, padding = filter_specs

            # Pad like in tensorflow's SAME/VALID mode.
            if padding == "same":
                padding_size, out_size = same_padding(in_size, kernel_size, strides)
                layers.append(nn.ZeroPad2d(padding_size))
            # No actual padding is performed for "valid" mode, but we will still
            # compute the output size (input for the next layer).
            else:
                out_size = valid_padding(in_size, kernel_size, strides)

            layer = nn.Conv2d(
                in_depth, out_depth, kernel_size, strides, bias=cnn_use_bias
            )

            # Initialize CNN layer kernel if necessary.
            if cnn_kernel_initializer:
                cnn_kernel_initializer(
                    layer.weight, **cnn_kernel_initializer_config or {}
                )
            # Initialize CNN layer bias if necessary.
            if cnn_bias_initializer:
                cnn_bias_initializer(layer.bias, **cnn_bias_initializer_config or {})

            layers.append(layer)

            # Layernorm.
            if cnn_use_layernorm:
                # We use an epsilon of 0.001 here to mimick the Tf default behavior.
                layers.append(LayerNorm1D(out_depth, eps=0.001))
            # Activation.
            if cnn_activation is not None:
                layers.append(cnn_activation())

            in_size = out_size
            in_depth = out_depth

        # Create the CNN.
        self.cnn = nn.Sequential(*layers)

    def forward(self, inputs):
        # Permute b/c data comes in as channels_last ([B, dim, dim, channels]) ->
        # Convert to `channels_first` for torch:
        inputs = inputs.permute(0, 3, 1, 2)
        out = self.cnn(inputs)
        # Permute back to `channels_last`.
        return out.permute(0, 2, 3, 1)


class LayerNorm1D(nn.Module):
    def __init__(self, num_features, **kwargs):
        super().__init__()
        self.layer_norm = nn.LayerNorm(num_features, **kwargs)

    def forward(self, x):
        # x shape: (B, dim, dim, channels).
        batch_size, channels, h, w = x.size()
        # Reshape to (batch_size * height * width, channels) for LayerNorm
        x = x.permute(0, 2, 3, 1).reshape(-1, channels)
        # Apply LayerNorm
        x = self.layer_norm(x)
        # Reshape back to (batch_size, dim, dim, channels)
        x = x.reshape(batch_size, h, w, channels).permute(0, 3, 1, 2)
        return x
