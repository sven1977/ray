import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core import Columns, DEFAULT_MODULE_ID
from ray.rllib.core.models.configs import MLPHeadConfig
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
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
from ray.tune.registry import register_env

torch, nn = try_import_torch()

# Register the Minigrid env we want to train on.
register_env(
    "mini_grid",

    lambda cfg: ImgObsWrapper(
        gym.make("MiniGrid-DoorKey-8x8-v0", render_mode="rgb_array")
    ),
)

parser = add_rllib_example_script_args(
    default_reward=float("inf"), default_iters=50000, default_timesteps=10000000000
)
parser.set_defaults(enable_new_api_stack=True)


class MiniGridCNNEncoder(nn.Module):
    def __init__(self, observation_space, feature_dim):
        super().__init__()

        n_input_channels = observation_space.shape[0]

        self._cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (1, 1)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self._cnn(torch.as_tensor(
                observation_space.sample()[None]
            ).float()).shape[1]

        self._features = nn.Sequential(nn.Linear(n_flatten, feature_dim), nn.ReLU())

    def forward(self, obs, **kwargs):
        return self._features(self._cnn(obs.float()))


class MiniGridTorchRLModule(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        super().setup()

        cfg = self.config.model_config_dict
        feature_dim = cfg.get("feature_dim", 288)

        # Shared value- and policy encoder.
        self._encoder = MiniGridCNNEncoder(self.config.observation_space, feature_dim)

        # Policy head.
        self._policy_head = MLPHeadConfig(
            input_dims=[feature_dim],
            hidden_layer_dims=cfg["fcnet_hiddens"],
            hidden_layer_activation=cfg["fcnet_activation"],
            output_layer_dim=int(self.config.action_space.n),
            output_layer_activation="linear",
        ).build(framework="torch")

        # Value head.
        self._value_head = MLPHeadConfig(
            input_dims=[feature_dim],
            hidden_layer_dims=cfg.get("fcnet_hiddens_vf", cfg["fcnet_hiddens"]),
            hidden_layer_activation=cfg.get(
                "fcnet_activation_vf", cfg["fcnet_activation"]
            ),
            output_layer_dim=1,
            output_layer_activation="linear",
        ).build(framework="torch")

    def _forward_inference(self, batch, **kwargs):
        # Encoder forward pass.
        encoder_outs = self._encoder(batch[Columns.OBS])
        # Policy head.
        return {Columns.ACTION_DIST_INPUTS: self._policy_head(encoder_outs)}

    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch)

    def _forward_train(self, batch, **kwargs):
        output = {}

        # Encoder forward pass.
        encoder_outs = self._encoder(batch[Columns.OBS])

        # Value function forward pass.
        vf_out = self._value_head(encoder_outs)
        # Squeeze out last dim (value function node).
        output[Columns.VF_PREDS] = vf_out.squeeze(-1)

        # Policy head.
        action_logits = self._policy_head(encoder_outs)
        output[Columns.ACTION_DIST_INPUTS] = action_logits

        return output

    def compute_values(self, batch):
        encoder_outs = self._encoder(batch[Columns.OBS])
        # Value function forward pass.
        vf_out = self._value_head(encoder_outs)
        # Squeeze out last dim (value function node).
        return vf_out.squeeze(-1)

    def get_inference_action_dist_cls(self):
        return TorchCategorical

    def get_exploration_action_dist_cls(self):
        return TorchCategorical

    def get_train_action_dist_cls(self):
        return TorchCategorical


if __name__ == "__main__":
    args = parser.parse_args()

    base_config = (
        PPOConfig()
        .environment("mini_grid")
        .env_runners(
            num_envs_per_env_runner=5,
        )
        .training(
            # Plug in the correct Learner class.
            learner_class=PPOTorchLearnerWithCuriosity,
            num_sgd_iter=6,
            train_batch_size_per_learner=2000,
            lr=0.0003,
            vf_loss_coeff=10.0,
            learner_config_dict={
                # Intrinsic reward coefficient.
                "intrinsic_reward_coeff": 0.01,
                # Forward loss weight (vs inverse dynamics loss). Total ICM loss is:
                # L(total ICM) = (
                #     `forward_loss_weight` * L(forward)
                #     + (1.0 - `forward_loss_weight`) * L(inverse_dyn)
                # )
                "forward_loss_weight": 0.2,
            },
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                module_specs={
                    # The "main" RLModule (policy) to be trained by our algo.
                    DEFAULT_MODULE_ID: RLModuleSpec(
                        module_class=MiniGridTorchRLModule,
                        model_config_dict={
                            # Size of the feature vector coming out of the CNN encoder.
                            # Note that this CNN encoder is a different network than
                            # the one used in the `IntrinsicCuriosityModel` below, so
                            # you can use a different `feature_dim` value here.
                            "feature_dim": 256,
                            # Policy head.
                            "fcnet_hiddens": [512],
                            "fcnet_activation": "relu",
                            # Value head.
                            "fcnet_hiddens_vf": [512],
                            "fcnet_activation_vf": "relu",
                        }
                    ),
                    # The intrinsic curiosity model.
                    ICM_MODULE_ID: RLModuleSpec(
                        module_class=IntrinsicCuriosityModel,
                        # Only create the ICM on the Learner workers, NOT on the
                        # EnvRunners.
                        learner_only=True,
                        # Configure the architecture of the ICM here.
                        model_config_dict={
                            "feature_dim": 288,
                            # Provide the feature net (encoder) class here instead
                            # of using the default feature net (FCNet). An FCNet
                            # wouldn't work here b/c the observation space is an image.
                            "feature_net_nn_module_class": MiniGridCNNEncoder,
                            # Configure the other two networks: inverse- and forward
                            # nets. Both use the feature vector(s) coming out of the
                            # feature net their input.
                            "inverse_net_hiddens": (256, 256),
                            "inverse_net_activation": "relu",
                            "forward_net_hiddens": (256, 256),
                            "forward_net_activation": "relu",
                        },
                    ),
                },
            ),
            # Use a different learning rate for training the ICM.
            algorithm_config_overrides_per_module={
                ICM_MODULE_ID: PPOConfig.overrides(lr=0.0001)
            },
        )
    )

    run_rllib_example_script_experiment(base_config, args)
