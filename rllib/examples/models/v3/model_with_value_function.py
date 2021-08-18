import numpy as np

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


class RNNModel(tf.keras.models.Model if tf else object):
    """An RNN-based model without a value function branch."""

    def __init__(self,
                 input_space,
                 *,
                 action_space,
                 num_outputs,
                 name="",
                 hiddens_size=256,
                 cell_size=64):
        super().__init__(name=name)

        self.cell_size = cell_size

        # Preprocess observation with a hidden layer and send to LSTM cell
        self.dense = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="dense1")
        self.lstm = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")

        # Postprocess LSTM output with another hidden layer and compute
        # values.
        self.logits = tf.keras.layers.Dense(
            num_outputs, activation=tf.keras.activations.linear, name="logits")

        #self.view_requirements = {}
        #self.view_requirements["state_in_0"] = ViewRequirement(
        #    data_col="state_out_0",
        #)

    @override("tf.keras.Model")
    def call(self, sample_batch):
        dense_out = self.dense(sample_batch[SampleBatch.OBS])
        B = tf.shape(sample_batch[SampleBatch.SEQ_LENS])[0]
        lstm_in = tf.reshape(dense_out, [B, -1, dense_out.shape.as_list()[1]])
        lstm_out, h, c = self.lstm(
            inputs=lstm_in,
            mask=tf.sequence_mask(sample_batch["seq_lens"]),
            initial_state=[
                sample_batch["state_in_0"], sample_batch["state_in_1"]
            ],
        )
        lstm_out = tf.reshape(lstm_out, [-1, lstm_out.shape.as_list()[2]])
        return lstm_out, [h, c]

    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]


class RNNModelWithValueFunction(tf.keras.models.Model if tf else object):
    """An RNN-based model with a value function branch."""

    def __init__(self,
                 input_space,
                 *,
                 action_space,
                 num_outputs,
                 name="",
                 hiddens_size=256,
                 cell_size=64):
        super().__init__(name=name)

        self.cell_size = cell_size

        # Preprocess observation with a hidden layer and send to LSTM cell
        self.dense = tf.keras.layers.Dense(
            hiddens_size, activation=tf.nn.relu, name="dense1")
        self.lstm = tf.keras.layers.LSTM(
            cell_size, return_sequences=True, return_state=True, name="lstm")

        # Postprocess LSTM output with another hidden layer and compute
        # values.
        self.logits = tf.keras.layers.Dense(
            num_outputs, activation=tf.keras.activations.linear, name="logits")
        self.values = tf.keras.layers.Dense(1, activation=None, name="values")

        #self.view_requirements = {}
        #self.view_requirements["state_in_0"] = ViewRequirement(
        #    data_col="state_out_0",
        #)

    @override("tf.keras.Model")
    def call(self, sample_batch):
        return self.forward_policy(sample_batch)

    def _forward_shared(self, sample_batch):
        dense_out = self.dense(sample_batch[SampleBatch.OBS])
        B = tf.shape(sample_batch[SampleBatch.SEQ_LENS])[0]
        lstm_in = tf.reshape(dense_out, [B, -1, dense_out.shape.as_list()[1]])
        lstm_out, h, c = self.lstm(
            inputs=lstm_in,
            mask=tf.sequence_mask(sample_batch["seq_lens"]),
            initial_state=[
                sample_batch["state_in_0"], sample_batch["state_in_1"]
            ],
        )
        lstm_out = tf.reshape(lstm_out, [-1, lstm_out.shape.as_list()[2]])
        return lstm_out, h, c

    def forward_policy(self, sample_batch):
        lstm_out, h, c = self._forward_shared(sample_batch)
        logits = self.logits(lstm_out)
        return logits, [h, c]

    def forward_value(self, sample_batch):
        lstm_out, _, _ = self._forward_shared(sample_batch)
        values = tf.reshape(self.values(lstm_out), [-1])
        return values, []

    def forward_policy_and_value(self, sample_batch):
        lstm_out, h, c = self._forward_shared(sample_batch)
        logits = self.logits(lstm_out)
        values = tf.reshape(self.values(lstm_out), [-1])
        return (logits, values), [h, c]

    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]
