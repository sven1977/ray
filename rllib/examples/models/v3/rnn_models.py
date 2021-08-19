import numpy as np

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class RNNModel(tf.keras.models.Model if tf else object):
    """An RNN-based model without a value function branch.

    It's simple structure is:
    input -> Linear+ReLU() -> LSTM() -> Linear() -> output, [h, c]
                                     -> [h, c]   /
    """

    def __init__(self,
                 input_space,
                 *,
                 output_size,
                 name="",
                 hiddens_size=256,
                 cell_size=64,
                 **kwargs,
                 ):
        """Initializes a RNNModel instance.

        Args:
            input_space (gym.Space): The input space for this model.
            output_size (int): The number of nodes in the last dense layer.
            name (str): An optional name for this keras model.
            hiddens_size (int): The size of the first dense layer.
            cell_size (int): The size of the LSTM layer.
        """
        super().__init__(name=name)

        self.cell_size = cell_size

        # First dense layer (ReLU).
        self.dense = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="dense1")
        # LSTM layer.
        self.lstm = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")

        # Output layer (linear).
        self.outputs = tf.keras.layers.Dense(
            output_size, activation=tf.keras.activations.linear, name="outputs")

    @override("tf.keras.Model")
    def call(self, sample_batch):
        """Defines the forward logic of this Model.

        Args:
            sample_batch (SampleBatch): The input dict (SampleBatch).
        """
        # Get the first layer's output (passing in OBS).
        dense_out = self.dense(sample_batch[SampleBatch.OBS])
        # Get the batch size by simply checking how many sequences we have.
        B = tf.shape(sample_batch[SampleBatch.SEQ_LENS])[0]
        # Add time axis before sending to LSTM.
        lstm_in = tf.reshape(dense_out, [B, -1, dense_out.shape.as_list()[1]])
        # Do the LSTM forward pass.
        lstm_out, h, c = self.lstm(
            inputs=lstm_in,
            mask=tf.sequence_mask(sample_batch["seq_lens"]),
            initial_state=[
                sample_batch["state_in_0"], sample_batch["state_in_1"]
            ],
        )
        # Remove the time axis again.
        lstm_out = tf.reshape(lstm_out, [-1, lstm_out.shape.as_list()[2]])
        # Pass through final layer.
        outputs = self.outputs(lstm_out)
        return outputs, [h, c]

    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]


class RNNModelWithValueFunction(RNNModel):
    """An RNN-based model with a value function branch."""

    def __init__(self,
                 input_space,
                 *,
                 logits_size,
                 name="",
                 hiddens_size=256,
                 cell_size=64,
                 **kwargs,
                 ):
        """Initializes a RNNModelWithValueFunction instance.

        Args:
            input_space (gym.Space): The input space for this model.
            logits_size (int): The number of nodes in the last dense layer.
            name (str): An optional name for this keras model.
            hiddens_size (int): The size of the first dense layer.
            cell_size (int): The size of the LSTM layer.
        """
        super().__init__(
            input_space,
            output_size=logits_size,
            hiddens_size=hiddens_size,
            cell_size=cell_size,
            name=name,
            **kwargs,
        )

        # The single value output node.
        self.values = tf.keras.layers.Dense(1, activation=None, name="values")

    def _shared_branch(self, sample_batch):
        # Get the first layer's output (passing in OBS).
        dense_out = self.dense(sample_batch[SampleBatch.OBS])
        # Get the batch size by simply checking how many sequences we have.
        B = tf.shape(sample_batch[SampleBatch.SEQ_LENS])[0]
        # Add time axis before sending to LSTM.
        lstm_in = tf.reshape(dense_out, [B, -1, dense_out.shape.as_list()[1]])
        # Do the LSTM forward pass.
        lstm_out, h, c = self.lstm(
            inputs=lstm_in,
            mask=tf.sequence_mask(sample_batch["seq_lens"]),
            initial_state=[
                sample_batch["state_in_0"], sample_batch["state_in_1"]
            ],
        )
        # Remove the time axis again.
        lstm_out = tf.reshape(lstm_out, [-1, lstm_out.shape.as_list()[2]])
        return lstm_out, h, c

    def policy(self, sample_batch):
        return self.__call__(sample_batch)

    def value(self, sample_batch):
        lstm_out, _, _ = self._shared_branch(sample_batch)
        values = tf.reshape(self.values(lstm_out), [-1])
        return values, []

    def policy_and_value(self, sample_batch):
        lstm_out, h, c = self._shared_branch(sample_batch)
        logits = self.outputs(lstm_out)
        values = tf.reshape(self.values(lstm_out), [-1])
        return (logits, values), [h, c]
