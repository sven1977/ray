from typing import Any, Dict

import gymnasium as gym
import torch

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.columns import Columns
from ray.rllib.core.models.torch.primitives import TorchCNN
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack
from ray.rllib.utils.annotations import override
from ray.rllib.utils.test_utils import add_rllib_example_script_args

nn = torch.nn


class MyAtariCNN(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        cfg = self.config.model_config_dict

        self._cnn = nn.Sequential(
            TorchCNN(
                input_dims=self.config.observation_space.shape,
                cnn_filter_specifiers=cfg["conv_filters"],
            ),
            nn.Flatten(),
        )
        # Compute feature dim by doing a dummy forward pass.
        with torch.no_grad():
            dummy_obs = torch.as_tensor(self.config.observation_space.sample()[None])
            feature_dim = self._cnn(dummy_obs).shape[1]
        self._pi_head = nn.Sequential(
            nn.Linear(feature_dim, cfg["policy_fcnet_hiddens"]),
            nn.ReLU(),
            nn.Linear(cfg["policy_fcnet_hiddens"], self.config.action_space.n),
        )
        self._values = nn.Sequential(
            nn.Linear(feature_dim, cfg["vf_fcnet_hiddens"]),
            nn.ReLU(),
            nn.Linear(cfg["vf_fcnet_hiddens"], 1),
        )

    @override(TorchRLModule)
    def _forward_inference(self, batch, **kwargs):
        # CNN stack (plus flatten).
        features = self._cnn(batch[Columns.OBS])
        # Policy head.
        logits = self._pi_head(features)
        return {Columns.ACTION_DIST_INPUTS: logits}

    @override(TorchRLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        # CNN stack (plus flatten).
        features = self._cnn(batch[Columns.OBS])
        # Policy head.
        logits = self._pi_head(features)
        # Values.
        values = self._values(features).squeeze(-1)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: values,
        }

    # We implement this RLModule as a ValueFunctionAPI RLModule, so it can be used
    # by value-based methods like PPO or IMPALA.
    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any]):
        # CNN stack (plus flatten).
        features = self._cnn(batch[Columns.OBS])
        # Values.
        return self._values(features).squeeze(-1)


parser = add_rllib_example_script_args(
    default_reward=float("inf"),
    default_timesteps=3000000,
    default_iters=100000000000,
)
parser.set_defaults(
    enable_new_api_stack=True,
    env="ALE/Pong-v5",
)
# Use `parser` to add your own custom command line options to this script
# and (if needed) use their values toset up `config` below.
args = parser.parse_args()


# Create a custom Atari setup (w/o the usual RLlib-hard-coded framestacking in it).
# We would like our frame stacking connector to do this job.
def _env_creator(cfg):
    return wrap_atari_for_new_api_stack(
        gym.make(
            args.env,
            # Make analogous to old v4 + NoFrameskip.
            frameskip=1,
            full_action_space=False,
            repeat_action_probability=0.0,
        ),
        framestack=4,
    )


tune.register_env("env", _env_creator)


config = (
    PPOConfig()
    .environment("env")
    .training(
        train_batch_size_per_learner=4000,  # 5000 on old yaml example
        minibatch_size=128,  # 500 on old yaml example
        num_epochs=10,
        lambda_=0.95,
        kl_coeff=0.5,
        clip_param=0.1,
        vf_clip_param=10.0,
        entropy_coeff=0.01,
        lr=0.00015 * args.num_gpus,
        grad_clip=100.0,
        grad_clip_by="global_norm",
    )
    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=MyAtariCNN,
            model_config_dict={
                "conv_filters": [
                    # num filters, kernel, stride
                    [16, 4, 2],
                    [32, 4, 2],
                    [64, 4, 2],
                    [128, 4, 2],
                ],
                "policy_fcnet_hiddens": 256,
                "vf_fcnet_hiddens": 256,
            },
        ),
    )
)


if __name__ == "__main__":
    from ray.rllib.utils.test_utils import run_rllib_example_script_experiment

    run_rllib_example_script_experiment(config, args=args)
