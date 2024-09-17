import gymnasium as gym
from minigrid.wrappers import FullyObsWrapper
import numpy as np

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core import Columns, DEFAULT_MODULE_ID
from ray.rllib.core.models.configs import MLPHeadConfig
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.examples.envs.env_rendering_and_recording import EnvRenderCallback
from ray.rllib.examples.learners.classes.intrinsic_curiosity_learners import (
    ICM_MODULE_ID,
    PPOTorchLearnerWithCuriosity
)
from ray.rllib.examples.rl_modules.classes.intrinsic_curiosity_model_rlm import (
    IntrinsicCuriosityModel,
)
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)

torch, nn = try_import_torch()

parser = add_rllib_example_script_args(
    default_reward=float("inf"), default_iters=50000, default_timesteps=10000000000
)
parser.set_defaults(
    enable_new_api_stack=True,
    env="MiniGrid-MemoryS7-v0",
)


class MiniGridCNNEncoder(nn.Module):
    def __init__(self, observation_space, feature_dim):
        super().__init__()

        n_input_channels = observation_space["image"].shape[2]

        self._cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2), 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2), 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2), 1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            dummy_obs = torch.as_tensor(observation_space["image"].sample()[None])
            n_flatten = self._cnn(dummy_obs.float().permute((0, 3, 1, 2))).shape[1]

        # +4: Add direction to feature vector as one-hot.
        self._features = nn.Sequential(
            nn.Linear(n_flatten + 4, feature_dim),
            nn.ReLU(),
        )

    def forward(self, batch, **kwargs):
        images = batch[Columns.OBS]["image"].float() * 10.0
        direction = batch[Columns.OBS]["direction"]

        # Fold time-ranks.
        B, T = images.shape[:2]
        images = images.reshape([-1] + list(images.shape[2:]))
        direction = direction.reshape([-1] + list(direction.shape[2:]))

        # Make channels first.
        images = images.permute((0, 3, 1, 2))

        cnn_out = self._cnn(images)
        direction = nn.functional.one_hot(direction, num_classes=4).float()
        in_ = torch.concat([cnn_out, direction], -1)

        features = self._features(in_)
        # Unfold time rank.
        features = features.reshape([B, T, features.shape[-1]])

        return {"features": features}


class MiniGridTorchRLModule(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        super().setup()

        cfg = self.config.model_config_dict
        feature_dim = cfg.get("feature_dim", 256)
        self._lstm_cell_size = cfg.get("lstm_cell_size", 256)

        # Shared value- and policy encoder.
        # Encode a flat one-hot vector (flat concatenation of all "pixels").
        # Use a CNN encoder.
        self._encoder = MiniGridCNNEncoder(
            self.config.observation_space, feature_dim
        )

        self._lstm = nn.LSTM(feature_dim, self._lstm_cell_size, batch_first=True)

        # Policy head.
        self._policy_head = MLPHeadConfig(
            input_dims=[self._lstm_cell_size],
            hidden_layer_dims=cfg["fcnet_hiddens"],
            hidden_layer_activation=cfg["fcnet_activation"],
            output_layer_dim=int(self.config.action_space.n),
            output_layer_activation="linear",
        ).build(framework="torch")

        # Value head.
        self._value_head = MLPHeadConfig(
            input_dims=[self._lstm_cell_size],
            hidden_layer_dims=cfg.get("fcnet_hiddens_vf", cfg["fcnet_hiddens"]),
            hidden_layer_activation=cfg.get(
                "fcnet_activation_vf", cfg["fcnet_activation"]
            ),
            output_layer_dim=1,
            output_layer_activation="linear",
        ).build(framework="torch")

    def get_initial_state(self):
        lstm_size = self._lstm_cell_size
        return {
            "h": np.zeros(shape=(lstm_size,), dtype=np.float32),
            "c": np.zeros(shape=(lstm_size,), dtype=np.float32),
        }

    def _forward_inference(self, batch, **kwargs):
        # Encoder forward pass.
        encoder_outs = self._encoder(batch)
        features = encoder_outs.pop("features")
        # LSTM.
        state_in = batch[Columns.STATE_IN]
        h, c = state_in["h"], state_in["c"]
        lstm_out, (h, c) = self._lstm(
            features,
            (h.unsqueeze(0), c.unsqueeze(0)),
        )
        # Policy head.
        ret = {
            Columns.ACTION_DIST_INPUTS: self._policy_head(lstm_out),
            Columns.STATE_OUT: {"h": h.squeeze(0), "c": c.squeeze(0)},
        }
        ret.update(encoder_outs)
        return ret

    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch)

    def _forward_train(self, batch, **kwargs):
        output = {}

        # Encoder forward pass.
        encoder_outs = self._encoder(batch)
        features = encoder_outs.pop("features")

        # LSTM.
        state_in = batch[Columns.STATE_IN]
        h, c = state_in["h"], state_in["c"]
        lstm_out, (h, c) = self._lstm(
            features,
            (h.unsqueeze(0), c.unsqueeze(0)),
        )
        output[Columns.STATE_OUT] = {"h": h.squeeze(0), "c": c.squeeze(0)}

        # Value function forward pass.
        vf_out = self._value_head(lstm_out)
        # Squeeze out last dim (value function node).
        output[Columns.VF_PREDS] = vf_out.squeeze(-1)

        # Policy head.
        action_logits = self._policy_head(lstm_out)
        output[Columns.ACTION_DIST_INPUTS] = action_logits

        output.update(encoder_outs)

        return output

    def compute_values(self, batch):
        encoder_outs = self._encoder(batch)
        features = encoder_outs.pop("features")
        # LSTM.
        state_in = batch[Columns.STATE_IN]
        h, c = state_in["h"], state_in["c"]
        lstm_out, _ = self._lstm(
            features,
            (h.unsqueeze(0), c.unsqueeze(0)),
        )
        # Value function forward pass.
        vf_out = self._value_head(lstm_out)
        # Squeeze out last dim (value function node).
        return vf_out.squeeze(-1)

    def get_inference_action_dist_cls(self):
        return TorchCategorical

    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    def get_train_action_dist_cls(self):
        return TorchCategorical


class ImgDirectionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        original_space = self.env.observation_space
        self.observation_space = gym.spaces.Dict({
            key: original_space[key] for key in original_space.spaces
            if key != 'mission'
        })

    def observation(self, observation):
        obs = observation.copy()
        obs.pop("mission")
        return obs


if __name__ == "__main__":
    args = parser.parse_args()

    # Register the Minigrid env we want to train on.
    tune.register_env(
        "mini_grid",
        lambda cfg: ImgDirectionWrapper(gym.make(args.env, render_mode="rgb_array")),
    )

    base_config = (
        PPOConfig()
        .environment("mini_grid")
        .env_runners(
            num_envs_per_env_runner=5,
        )
        .training(
            # Plug in the correct Learner class.
            num_sgd_iter=6,
            train_batch_size_per_learner=2000,
            lr=0.0003,
            vf_loss_coeff=1.0,
            entropy_coeff=0.05,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=MiniGridTorchRLModule,
                model_config_dict={
                    "max_seq_len": 20,
                    # Use the flat one-hot encoder (or the CNN encoder).
                    #"one_hot_encoder": args.one_hot_encoder,
                    # If cell size >0 -> use an LSTM.
                    "lstm_cell_size": 256,
                    # Size of the feature vector coming out of the CNN encoder.
                    # Note that this CNN encoder is a different network than
                    # the one used in the `IntrinsicCuriosityModel` below, so
                    # you can use a different `feature_dim` value here.
                    "feature_dim": 256,
                    # Policy head.
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                    # Value head.
                    "fcnet_hiddens_vf": [256, 256],
                    "fcnet_activation_vf": "relu",
                }
            ),
        )
    )

    run_rllib_example_script_experiment(base_config, args)
