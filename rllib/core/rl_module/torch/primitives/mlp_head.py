from typing import Callable, Dict, List, Optional, Union

from ray.rllib.core.rl_module.torch.primitives.mlp import TorchMLP
from ray.rllib.models.torch.torch_distributions import TorchDiagGaussian
from ray.rllib.models.utils import get_activation_fn, get_initializer_fn
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


def build_mlp_head(
    input_dim,
    model_config,
    *,
    action_dist_class = None,
    action_space = None,
    output_dim: int = None,
):
    is_diag_gaussian = False

    # Define the output dimension via the action distribution.
    if action_dist_class is not None:
        is_diag_gaussian = issubclass(action_dist_class, TorchDiagGaussian)
        if model_config["free_log_std"] and not is_diag_gaussian:
            raise ValueError(
                "If `free_log_std` is True, your action distribution "
                f"({action_dist_class}) must be of type `TorchDiagGaussian`!"
            )
        output_dim = action_dist_class.required_input_dim(
            space=action_space, model_config=model_config
        )
    return TorchMLPHead(
        input_dim=input_dim,
        model_config=model_config,
        output_dim=output_dim,
        clip_log_std=is_diag_gaussian,
    )
    # With the action distribution class and the number of outputs defined,
    # we can build the config for the policy head.
    # pi_head_cls = (
    #    FreeLogStdMLPHeadConfig
    #    if self.model_config["free_log_std"]
    #    else MLPHeadConfig
    # )


class TorchMLPHead(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_config: dict,
        output_dim: int,
        clip_log_std: bool,
    ):
        super().__init__()

        self._half_output_dim = output_dim // 2

        self._free_log_std = model_config["free_log_std"]
        if self._free_log_std:
            assert output_dim % 2 == 0, "output_dims must be even for free std!"
            self._log_std = torch.nn.Parameter(
                torch.as_tensor([0.0] * self._half_output_dim)
            )

        self._net = TorchMLP(
            input_dim=input_dim,
            hidden_layer_dims=model_config["head_fcnet_hiddens"],
            hidden_layer_activation=model_config["head_fcnet_activation"],
            #hidden_layer_use_layernorm=config.hidden_layer_use_layernorm,
            #hidden_layer_use_bias=config.hidden_layer_use_bias,
            hidden_layer_weights_initializer=(
                model_config["head_fcnet_kernel_initializer"]
            ),
            hidden_layer_weights_initializer_config=(
                model_config["head_fcnet_kernel_initializer_kwargs"]
            ),
            hidden_layer_bias_initializer=model_config["head_fcnet_bias_initializer"],
            hidden_layer_bias_initializer_config=(
                model_config["head_fcnet_bias_initializer_kwargs"]
            ),
            output_dim=self._half_output_dim if self._free_log_std else output_dim,
            output_activation="linear",
            #output_use_bias=config.output_layer_use_bias,
            # TODO (sven): Does the output layer need its own initialization settings?
            output_weights_initializer=model_config["head_fcnet_kernel_initializer"],
            output_weights_initializer_config=(
                model_config["head_fcnet_kernel_initializer_kwargs"]
            ),
            output_bias_initializer=model_config["head_fcnet_bias_initializer"],
            output_bias_initializer_config=(
                model_config["head_fcnet_bias_initializer_kwargs"]
            ),
        )
        # If log standard deviations should be clipped. This should be only true for
        # policy heads. Value heads should never be clipped.
        self._clip_log_std = clip_log_std
        if self._clip_log_std:
            # The clipping parameter for the log standard deviation.
            self._log_std_clip_param = torch.Tensor(
                [model_config["log_std_clip_param"]]
            )
            # Register a buffer to handle device mapping.
            self.register_buffer("log_std_clip_param_const", self._log_std_clip_param)

    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        # Forward pass.
        output = self._net(inputs)

        # Clip the log std, if necessary.
        if self._clip_log_std:
            if self._free_log_std:
                mean = output
                log_std = self._log_std
            else:
                mean, log_std = torch.chunk(output, chunks=2, dim=-1)
            log_std = torch.clamp(
                log_std,
                -self._log_std_clip_param_const,
                self._log_std_clip_param_const,
            )
            if self._free_log_std:
                return torch.cat(
                    [mean, log_std.unsqueeze(0).repeat([len(mean), 1])], dim=1
                )
            else:
                return torch.cat((mean, log_std), dim=-1)
        else:
            if self._free_log_std:
                return torch.cat(
                    [output, self._log_std.unsqueeze(0).repeat([len(output), 1])], dim=1
                )
            else:
                return output
