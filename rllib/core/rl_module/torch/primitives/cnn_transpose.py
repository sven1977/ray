from typing import Callable, Dict, List, Optional, Union, Tuple

from ray.rllib.core.rl_module.torch.primitives.cnn import LayerNorm1D
from ray.rllib.core.rl_module.torch.utils import Stride2D
from ray.rllib.models.torch.misc import same_padding_transpose_after_stride
from ray.rllib.models.utils import get_activation_fn, get_initializer_fn
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class TorchCNNTranspose(nn.Module):
    """A model containing a CNNTranspose with N Conv2DTranspose layers.

    All layers share the same activation function, bias setup (use bias or not),
    and LayerNormalization setup (use layer normalization or not), except for the last
    one, which is never activated and never layer norm'd.

    Note that there is no reshaping/flattening nor an additional dense layer at the
    beginning or end of the stack. The input as well as output of the network are 3D
    tensors of dimensions [width x height x num output filters].
    """

    def __init__(
        self,
        *,
        input_dims: Union[List[int], Tuple[int]],
        cnn_transpose_filter_specifiers: List[List[Union[int, List]]],
        cnn_transpose_use_bias: bool = True,
        cnn_transpose_activation: str = "relu",
        cnn_transpose_use_layernorm: bool = False,
        cnn_transpose_kernel_initializer: Optional[Union[str, Callable]] = None,
        cnn_transpose_kernel_initializer_config: Optional[Dict] = None,
        cnn_transpose_bias_initializer: Optional[Union[str, Callable]] = None,
        cnn_transpose_bias_initializer_config: Optional[Dict] = None,
    ):
        """Initializes a TorchCNNTranspose instance.

        Args:
            input_dims: The 3D input dimensions of the network (incoming image).
            cnn_transpose_filter_specifiers: A list of lists, where each item represents
                one Conv2DTranspose layer. Each such Conv2DTranspose layer is further
                specified by the elements of the inner lists. The inner lists follow
                the format: `[number of filters, kernel, stride]` to
                specify a convolutional-transpose layer stacked in order of the
                outer list.
                `kernel` as well as `stride` might be provided as width x height tuples
                OR as single ints representing both dimension (width and height)
                in case of square shapes.
            cnn_transpose_use_bias: Whether to use bias on all Conv2DTranspose layers.
            cnn_transpose_use_layernorm: Whether to insert a LayerNormalization
                functionality in between each Conv2DTranspose layer's outputs and its
                activation.
                The last Conv2DTranspose layer will not be normed, regardless.
            cnn_transpose_activation: The activation function to use after each layer
                (except for the last Conv2DTranspose layer, which is always
                non-activated).
            cnn_transpose_kernel_initializer: The initializer function or class to use
                for kernel initialization in the CNN layers. If `None` the default
                initializer of the respective CNN layer is used. Note, only the
                in-place initializers, i.e. ending with an underscore "_" are allowed.
            cnn_transpose_kernel_initializer_config: Configuration to pass into the
                initializer defined in `cnn_transpose_kernel_initializer`.
            cnn_transpose_bias_initializer: The initializer function or class to use for
                bias initialization in the CNN layers. If `None` the default initializer
                of the respective CNN layer is used. Note, only the in-place
                initializers, i.e. ending with an underscore "_" are allowed.
            cnn_transpose_bias_initializer_config: Configuration to pass into the
                initializer defined in `cnn_transpose_bias_initializer`.
        """
        super().__init__()

        assert len(input_dims) == 3

        cnn_transpose_activation = get_activation_fn(
            cnn_transpose_activation, framework="torch"
        )
        cnn_transpose_kernel_initializer = get_initializer_fn(
            cnn_transpose_kernel_initializer, framework="torch"
        )
        cnn_transpose_bias_initializer = get_initializer_fn(
            cnn_transpose_bias_initializer, framework="torch"
        )

        layers = []

        # Add user-specified hidden convolutional layers first
        width, height, in_depth = input_dims
        in_size = [width, height]
        for i, (out_depth, kernel, stride) in enumerate(
            cnn_transpose_filter_specifiers
        ):
            is_final_layer = i == len(cnn_transpose_filter_specifiers) - 1

            # Resolve stride and kernel width/height values if only int given (squared).
            s_w, s_h = (stride, stride) if isinstance(stride, int) else stride
            k_w, k_h = (kernel, kernel) if isinstance(kernel, int) else kernel

            # Stride the incoming image first.
            stride_layer = Stride2D(in_size[0], in_size[1], s_w, s_h)
            layers.append(stride_layer)
            # Then 0-pad (like in tensorflow's SAME mode).
            # This will return the necessary padding such that for stride=1, the output
            # image has the same size as the input image, for stride=2, the output image
            # is 2x the input image, etc..
            padding, out_size = same_padding_transpose_after_stride(
                (stride_layer.out_width, stride_layer.out_height), kernel, stride
            )
            layers.append(nn.ZeroPad2d(padding))  # left, right, top, bottom
            # Then do the Conv2DTranspose operation
            # (now that we have padded and strided manually, w/o any more padding using
            # stride=1).

            layer = nn.ConvTranspose2d(
                in_depth,
                out_depth,
                kernel,
                # Force-set stride to 1 as we already took care of it.
                1,
                # Disable torch auto-padding (torch interprets the padding setting
                # as: dilation (==1.0) * [`kernel` - 1] - [`padding`]).
                padding=(k_w - 1, k_h - 1),
                # Last layer always uses bias (b/c has no LayerNorm, regardless of
                # config).
                bias=cnn_transpose_use_bias or is_final_layer,
            )

            # Initialize CNN Transpose layer kernel if necessary.
            if cnn_transpose_kernel_initializer:
                cnn_transpose_kernel_initializer(
                    layer.weight, **cnn_transpose_kernel_initializer_config or {}
                )
            # Initialize CNN Transpose layer bias if necessary.
            if cnn_transpose_bias_initializer:
                cnn_transpose_bias_initializer(
                    layer.bias, **cnn_transpose_bias_initializer_config or {}
                )

            layers.append(layer)
            # Layernorm (never for final layer).
            if cnn_transpose_use_layernorm and not is_final_layer:
                layers.append(LayerNorm1D(out_depth, eps=0.001))
            # Last layer is never activated (regardless of config).
            if cnn_transpose_activation is not None and not is_final_layer:
                layers.append(cnn_transpose_activation())

            in_size = (out_size[0], out_size[1])
            in_depth = out_depth

        # Create the final CNNTranspose network.
        self.cnn_transpose = nn.Sequential(*layers)

    def forward(self, inputs):
        # Permute b/c data comes in as [B, dim, dim, channels]:
        out = inputs.permute(0, 3, 1, 2)
        out = self.cnn_transpose(out)
        return out.permute(0, 2, 3, 1)
