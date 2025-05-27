import gymnasium as gym
import numpy as np
import tree  # pip install dm_tree

from ray.rllib.core import Columns
from ray.rllib.core.rl_module.torch.primitives.cnn import TorchCNN
from ray.rllib.core.rl_module.torch.primitives.mlp import TorchMLP
from ray.rllib.models.utils import get_initializer_fn
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


def build_encoder(observation_space, model_config):

    if isinstance(observation_space, gym.spaces.Box):

        if len(observation_space.shape) == 1:
            encoder = TorchMLP(
                input_dim=observation_space.shape[0],
                hidden_layer_dims=model_config["fcnet_hiddens"],
                hidden_layer_activation=model_config["fcnet_activation"],
                hidden_layer_weights_initializer=(
                    model_config["fcnet_kernel_initializer"]
                ),
                hidden_layer_weights_initializer_config=(
                    model_config["fcnet_kernel_initializer_kwargs"]
                ),
                hidden_layer_bias_initializer=model_config["fcnet_bias_initializer"],
                hidden_layer_bias_initializer_config=(
                    model_config["fcnet_bias_initializer_kwargs"]
                ),
            )
            output_shape = (model_config["fcnet_hiddens"][-1],)

        elif len(observation_space.shape) == 3:
            encoder = nn.Sequential(
                TorchCNN(
                    input_dims=observation_space.shape,
                    cnn_filter_specifiers=model_config["conv_filters"],
                    cnn_activation=model_config["conv_activation"],
                    cnn_kernel_initializer=(
                        model_config["conv_kernel_initializer"]
                    ),
                    cnn_kernel_initializer_config=(
                        model_config["conv_kernel_initializer_kwargs"]
                    ),
                    cnn_bias_initializer=model_config["conv_bias_initializer"],
                    cnn_bias_initializer_config=(
                        model_config["conv_bias_initializer_kwargs"]
                    ),
                ),
                nn.Flatten(),
            )
            # Quick test forward pass to measure the output dim.
            test_out = encoder(
                torch.from_numpy(
                    np.random.random(size=(1,) + tuple(observation_space.shape))
                )
            )
            output_shape = (test_out.shape[-1],)

        else:
            raise ValueError(
                f"If `observation_space` ({observation_space}) is Box, its shape must "
                f"be 1D or 3D!"
            )
    else:
        raise ValueError(
            f"`observation_space` ({observation_space}) must be 1D or 3D Box!"
        )

    # Add LSTM layer(s) on top of the base encoder.
    if model_config["use_lstm"]:
        encoder = TorchLSTMEncoder(
            input_dim=output_shape[0],
            base_net=encoder,
            model_config=model_config,
        )
        output_shape = (model_config["lstm_cell_size"],)
    else:
        encoder = TorchEncoder(encoder)

    return encoder, output_shape


class TorchEncoder(nn.Module):
    def __init__(self, base_net):
        super().__init__()
        self._base_net = base_net

    def forward(self, inputs):
        return {Columns.EMBEDDINGS: self._base_net(inputs)}

    def get_initial_state(self):
        return {}


class TorchLSTMEncoder(nn.Module):

    def __init__(self, input_dim: int, model_config, base_net):
        super().__init__()
        self._base_net = base_net
        self._input_dim = input_dim
        self._model_config = model_config

        lstm_weights_initializer = get_initializer_fn(
            self._model_config["lstm_kernel_initializer"], framework="torch"
        )
        lstm_bias_initializer = get_initializer_fn(
            self._model_config["lstm_bias_initializer"], framework="torch"
        )

        # Create the torch LSTM layer.
        self._lstm = nn.LSTM(
            self._input_dim,
            self._model_config["lstm_cell_size"],
            1,  # num_layers
            batch_first=True,
            bias=True,
        )

        # Initialize LSTM layer weigths and biases, if necessary.
        for layer in self._lstm.all_weights:
            if lstm_weights_initializer:
                lstm_weights_initializer(
                    layer[0],
                    **self._model_config["lstm_kernel_initializer_kwargs"] or {},
                )
                lstm_weights_initializer(
                    layer[1],
                    **self._model_config["lstm_kernel_initializer_kwargs"] or {},
                )
            if lstm_bias_initializer:
                lstm_bias_initializer(
                    layer[2],
                    **self._model_config["lstm_bias_initializer_kwargs"] or {},
                )
                lstm_bias_initializer(
                    layer[3],
                    **self._model_config["lstm_bias_initializer_kwargs"] or {},
                )

    def forward(self, inputs: dict, **kwargs):
        outputs = {}

        # Push observations through the tokenizer encoder if we built one.
        out = self._base_net(inputs[Columns.OBS])

        # States are batch-first when coming in. Make them layers-first.
        states_in = tree.map_structure(
            lambda s: s.transpose(0, 1), inputs[Columns.STATE_IN]
        )

        out, states_out = self._lstm(out, (states_in["h"], states_in["c"]))
        states_out = {"h": states_out[0], "c": states_out[1]}

        # Insert them into the output dict.
        outputs[Columns.EMBEDDINGS] = out
        outputs[Columns.STATE_OUT] = tree.map_structure(
            lambda s: s.transpose(0, 1), states_out
        )
        return outputs

    def get_initial_state(self):
        return {
            "h": torch.zeros(1, self._model_config["lstm_cell_size"]),
            "c": torch.zeros(1, self._model_config["lstm_cell_size"]),
        }
