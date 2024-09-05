from ray.rllib.core import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env  # noqa

torch, nn = try_import_torch()


parser = add_rllib_example_script_args(
    default_reward=0.9, default_iters=50, default_timesteps=100000
)
parser.set_defaults(enable_new_api_stack=True)


class MinigridFeaturesExtractor(TorchRLModule, ValueFunctionAPI):
    def setup(self):
        super().setup()

        features_dim = self.config.model_config_dict.get("features_dim", 512)
        n_input_channels = self.config.observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(
                self.config.observation_space.sample()[None]
            ).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def _forward_inference(self, batch, **kwargs):
        obs = batch[Columns.OBS]
        logits = self.linear(self.cnn(obs))
        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }


if __name__ == "__main__":
    args = parser.parse_args()

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment("MiniGrid-Empty-8x8-v0")
        .rl_module(

        )
    )

    run_rllib_example_script_experiment(base_config, args)
