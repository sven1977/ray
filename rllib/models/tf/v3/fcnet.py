import numpy as np
import gym
from typing import Dict, Optional, Sequence, Union

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import TensorType, List, ModelConfigDict

tf1, tf, tfv = try_import_tf()


@ExperimentalAPI("ModelV3")
class FCNet(tf.keras.Model if tf else object):
    """Generic fully connected network implemented in tf Keras."""

    def __init__(
            self,
            input_space: gym.spaces.Space,
            *,
            action_space: gym.spaces.Space = None,
            name: str = "",
            fcnet_hiddens: Optional[Sequence[int]] = (),
            fcnet_activation: Optional[str] = None,
            output_layer_size: Optional[Union[str, int]] = None,
            output_layer_activation: Optional[str] = None,
            add_shared_vf_branch: bool = False,
            free_log_std: bool = False,
            **kwargs,
    ):
        super().__init__(name=name)

        # TODO: Not supported yet (keep it simple for now).
        assert add_shared_vf_branch is False and free_log_std is False
        assert isinstance(input_space, gym.spaces.Box)

        # If output_layer_size == "action_space": Derive output size
        # from given action space.
        if output_layer_size == "action_space":
            # TODO: only Box action_spaces supported so far
            #  (keep it simple for now).
            assert isinstance(action_space, gym.spaces.Box)
            output_layer_size = int(np.product(action_space.shape))

        activation = get_activation_fn(fcnet_activation)

        # We are using obs_flat, so take the flattened shape as input.
        inputs = tf.keras.layers.Input(
            shape=(int(np.product(input_space.shape)), ), name="inputs")

        # Last hidden layer output (before logits outputs).
        last_layer = inputs
        # Create all layers specified via `fcnet_hiddens`.
        for i, size in enumerate(fcnet_hiddens):
            last_layer = tf.keras.layers.Dense(
                size,
                name=f"fc_{i}",
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if output_layer_size:
            outputs = tf.keras.layers.Dense(
                output_layer_size,
                name="fc_out",
                activation=get_activation_fn(output_layer_activation),
                kernel_initializer=normc_initializer(1.0))(last_layer)
        else:
            outputs = last_layer

        self.base_model = tf.keras.Model(inputs, outputs)

    def call(self, input_dict: SampleBatch) -> \
            (TensorType, List[TensorType], Dict[str, TensorType]):
        outputs = self.base_model(input_dict[SampleBatch.OBS])
        return outputs, []
