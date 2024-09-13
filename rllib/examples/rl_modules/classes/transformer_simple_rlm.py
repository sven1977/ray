import math
from typing import Any, Dict

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()


class TransformerSimple(TorchRLModule, ValueFunctionAPI):
    """A simple transformer/attention net for value-function based RL algorithms.

    The architecture of this transformer model is simple.
    It contains a first linear layer to bring the assumed 1D observations into the
    "attention_dim" shape. We call these the "embeddings" even though no real Embedding
    layer is used (we don't have discrete tokens so there is no need for this).
    The "embeddings" are then scaled by the sqrt of their size and a positional
    encoding matrix is added.
    The resulting tokens are then pushed through n transformer (decoder-only) layers.
    The output of these layers is sent through a policy head to yield action logits
    and a value head to yield value estimates.

    The general shape of an input batch is (B, T, [obs]) and the general shape of
    the output is (B, T, [num actions]).

    Note the two types of masks used in this model:
    1) A "transformer_zero_padding" mask is required in case a batch
    contains rows stemming from an episode that is shorter than T. These rows will
    require right-zero-padding (masking), with 0.0 for valid values (not masked) and
    1.0 for masked (zero-padded) positions.
    This mask is required for inference in case of several episodes being stepped
    through at the same time (vectorization) and some of these episodes not having
    the exact same length. It is also used for training for chopping up the train
    batch into `max_seq_len` chunks (right-zero-padding the locations past an episode's
    terminal/truncation).
    2) A "causal_mask" is required for training only. It has a shape of (T, T), is non-
    batch-row specific (every batch row has the same causal mask), and used in order
    to not attend to timesteps in the future during training (which would be cheating
    and causing problems learning a decent policy).
    """

    @override(TorchRLModule)
    def setup(self):
        """Use this method to create all the model components that you require.

        Feel free to access the following useful properties in this class:
        - `self.config.model_config_dict`: The config dict for this RLModule class,
        which should contain flxeible settings, for example: {"hiddens": [256, 256]}.
        - `self.config.observation|action_space`: The observation and action space that
        this RLModule is subject to. Note that the observation space might not be the
        exact space from your env, but that it might have already gone through
        preprocessing through a connector pipeline (for example, flattening,
        frame-stacking, mean/std-filtering, etc..).
        """
        super().setup()

        assert len(self.config.observation_space.shape) == 1, (
            "Only supports 1D observation spaces!"
        )

        cfg = self.config.model_config_dict
        attention_dim = cfg.get("attention_dim", 256)
        attention_num_heads = cfg.get("attention_num_heads", 4)
        attention_num_transformer_units = cfg.get("attention_num_transformer_units", 1)
        max_seq_len = cfg.get("max_seq_len", 100)

        # Build the entire stack
        # Observations input layer mapping observation tensors to a unified 1D tensor
        # with shape=(attention_dim,). We call it embedding, b/c it takes the place of
        # an actual Embedding layer in a language model.
        self._embedding = nn.Linear(
            self.config.observation_space.shape[0], attention_dim
        )
        # Positional encoding layer.
        self._positional_encoding = PositionalEncoding(
            d_model=attention_dim, max_len=max_seq_len,
        )
        # The actual transformer block.
        decoder_layer = nn.TransformerDecoderLayer(attention_dim, attention_num_heads)
        self._transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            attention_num_transformer_units,
        )
        # The action logits output layer.
        self._logits = nn.Linear(attention_dim, self.config.action_space.n)
        # The value function head.
        self._values = nn.Linear(attention_dim, 1)

    @override(TorchRLModule)
    def _forward_inference(self, batch, **kwargs):
        # Compute the basic 1D feature tensor (inputs to policy- and value-heads).
        _, logits = self._compute_features_and_logits(batch)

        # TODO (sven): Maybe move this into a module-to-env connector?
        # Remove all logits except for the last one.
        sequence_lengths = batch["transformer_zero_padding"].bool().sum(dim=1)
        last_action_logits = logits[torch.arange(logits.size(0)), sequence_lengths - 1]

        # Return logits as ACTION_DIST_INPUTS (categorical distribution).
        return {Columns.ACTION_DIST_INPUTS: last_action_logits}

    @override(TorchRLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        # Compute the basic 1D feature tensor (inputs to policy- and value-heads).
        features, logits = self._compute_features_and_logits(batch, is_training=True)
        # Besides the action logits, we also have to return value predictions here
        # (to be used inside the loss function).
        values = self._values(features).squeeze(-1)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: values,
        }

    # We implement this RLModule as a ValueFunctionAPI RLModule, so it can be used
    # by value-based methods like PPO or IMPALA.
    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any]) -> TensorType:
        features = self._compute_features(batch, is_training=True)
        return self._values(features).squeeze(-1)

    def _compute_features_and_logits(self, batch, is_training=False):
        features = self._compute_features(batch, is_training=is_training)
        logits = self._logits(features)
        return features, logits

    def _compute_features(self, batch, is_training=False):
        obs = batch[Columns.OBS]
        assert len(obs.shape) == 3, (
            f"`obs` must have 3D shape (B, T, emb), but has shape {obs.shape}!"
        )
        #assert "causal_mask" in batch and len(batch["causal_mask"].shape) == 2
        embeddings = self._embedding(obs)
        embeddings *= math.sqrt(embeddings.size(-1))
        pos_encoded_embeddings = self._positional_encoding(embeddings)
        return self._transformer_decoder(
            pos_encoded_embeddings,
            pos_encoded_embeddings,
            tgt_mask=self._generate_causal_mask(embeddings.shape[1]),
            # Causal mask (do not use for inference).
            tgt_is_causal=True,#is_training,
            #tgt_mask=batch["causal_mask"],
            # Zero padding mask.
            # Note that `torch.nn.TransformerDecoder(memory_key_padding_mask=..)`
            # expects the mask to have 0.0 for valid values (unmasked) and 1.0 for
            # invalid/masked values. Hence, the need for the inversion operator (`~`).
            memory_key_padding_mask=~(batch["transformer_zero_padding"].bool()).transpose(0, 1)
        )

    # TODO (sven): In order for this RLModule to work with PPO, we must define
    #  our own `get_..._action_dist_cls()` methods. This would become more obvious,
    #  if we simply subclassed the `PPOTorchRLModule` directly here (which we didn't do
    #  for simplicity and to keep some generality). We might even get rid of algo-
    #  specific RLModule subclasses altogether in the future and replace them
    #  by mere algo-specific APIs (w/o any actual implementations).
    @override(RLModule)
    def get_train_action_dist_cls(self):
        return TorchCategorical

    @override(RLModule)
    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    @override(RLModule)
    def get_inference_action_dist_cls(self):
        return TorchCategorical

    def _generate_causal_mask(self, sz):
        """Create a causal mask of shape (T, T), where invalid positions have -inf.

        Note that the causal mask is always added directly to the attention scores so
        -inf means mask-out and 0.0 means mask-in (don't change).

        Use these masks only in the training setup (teacher-forced) for making sure that
        no position in the input is able to attend to future values (which would be
        "cheating" and bad for training the model).

        When passing the mask into the `torch.nn.TransformerDecoder()` call, use the
        `tgt_mask` arg.
        """
        # Create a triangular matrix with upper-right diagonal all -inf
        # (masked out/invalid) and the lower-left diagonal (including the main diagonal)
        # all 0.0 (masked in/valid).
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float("-inf"))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


if __name__ == "__main__":
    import gymnasium as gym
    import numpy as np

    obs_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
    act_space = gym.spaces.Discrete(2)
    attention_dim = 32
    attention_num_heads = 2
    attention_num_transformer_units = 1
    max_seq_len = 20

    model = TransformerSimple(
        RLModuleConfig(
            obs_space,
            act_space,
            model_config_dict={
                "attention_dim": attention_dim,
                "attention_num_heads": attention_num_heads,
                "attention_num_transformer_units": attention_num_transformer_units,
                "max_seq_len": max_seq_len,
            },
        )
    )

    # Example input.
    B = 2
    T = 10
    t_short_episode = 5  # <- the length of the shorter episode (the longer one is T).
    obs = torch.rand((B, T, obs_space.shape[0]))

    # Causal mask -> do NOT use one as we are doing inference here.
    # The shape of the causal mask is (T, T) and it is passed as:
    # `torch.nn.TransformerDecoder(tgt_mask=...)`
    # causal_mask = model._generate_causal_mask(T)

    # Zero-padding mask -> For zero'ing out timesteps past the episode's end.
    # Note that this is only necessary if we have a vector env with episodes that
    # have different lengths (and we need to right-zero-pad those episodes shorter than
    # the longest one in the vector).
    # The shape of the zero-padding mask is (B, T) and it is passed as:
    # `torch.nn.TransformerDecoder(memory_key_padding_mask=...)`
    zero_padding_mask = torch.tensor([
        [1.0] * T,
        [1.0] * t_short_episode + [0.0] * (T - t_short_episode)
    ])

    # Forward (inference) pass.
    output = model.forward_inference({
        Columns.OBS: obs,
        "transformer_zero_padding": zero_padding_mask,
    })
    print(output)