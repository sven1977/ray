# flake8: noqa
from ray.rllib.utils.annotations import override
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.core.models.specs.specs_base import TensorSpec


# __enabling-rlmodules-in-configs-begin__
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    # Enable the new API stack (and with it, the RLModule API).
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .framework("torch")
    .environment("CartPole-v1")
)
algo = config.build()

# Run 2 training iterations.
for _ in range(2):
    print(algo.train())
# __enabling-rlmodules-in-configs-end__


# __constructing-rlmodules-begin__
import gymnasium as gym
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule

# Create an env object to know the spaces.
env = gym.make("CartPole-v1")

# Construct an RLModuleSpec.
spec = RLModuleSpec(
    module_class=DiscreteBCTorchModule,
    observation_space=env.observation_space,
    action_space=env.action_space,
    # A custom dict that will be accessible inside your class as `self.model_config`.
    model_config={"fcnet_hiddens": [64]},
)

# Construct the actual RLModule object with .build():
rl_module = spec.build()
# __constructing-rlmodules-end__


# __constructing-multi-rlmodules-begin__
import gymnasium as gym
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule

# Construct a MultiRLModuleSpec.
spec = MultiRLModuleSpec(
    module_specs={
        "module_1": RLModuleSpec(
            module_class=DiscreteBCTorchModule,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(10,)),
            action_space=gym.spaces.Discrete(2),
            # A custom dict that will be accessible inside your class as
            # `self.model_config`.
            model_config={"fcnet_hiddens": [32]},
        ),
        "module_2": RLModuleSpec(
            module_class=DiscreteBCTorchModule,
            observation_space=gym.spaces.Box(low=-1, high=1, shape=(5,)),
            action_space=gym.spaces.Discrete(2),
            # A custom dict that will be accessible inside your class as
            # `self.model_config`.
            model_config={"fcnet_hiddens": [16]},
        ),
    },
)

# Construct the actual MultiRLModule object with .build():
multi_rl_module = spec.build()
# __constructing-multi-rlmodules-end__


# __pass-specs-to-configs-sa-begin__
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule
from ray.rllib.core.testing.bc_algorithm import BCConfigTest


config = (
    BCConfigTest()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment("CartPole-v1")
    .rl_module(
        model_config={"fcnet_hiddens": [32, 32]},
        rl_module_spec=RLModuleSpec(module_class=DiscreteBCTorchModule),
    )
)

algo = config.build()
print(algo.get_module())
# __pass-specs-to-configs-sa-end__


# __pass-specs-to-configs-ma-begin__
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule
from ray.rllib.core.testing.bc_algorithm import BCConfigTest
from ray.rllib.examples.envs.classes.multi_agent import MultiAgentCartPole


config = (
    BCConfigTest()
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )
    .environment(MultiAgentCartPole, env_config={"num_agents": 2})
    .rl_module(
        model_config_dict={"fcnet_hiddens": [32, 32]},
        rl_module_spec=MultiRLModuleSpec(
            module_specs=RLModuleSpec(module_class=DiscreteBCTorchModule)
        ),
    )
)
# __pass-specs-to-configs-ma-end__


# __convert-sa-to-ma-begin__
import gymnasium as gym
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.testing.torch.bc_module import DiscreteBCTorchModule

env = gym.make("CartPole-v1")
spec = RLModuleSpec(
    module_class=DiscreteBCTorchModule,
    observation_space=env.observation_space,
    action_space=env.action_space,
    model_config_dict={"fcnet_hiddens": [64]},
)

module = spec.build()
multi_rl_module = module.as_multi_rl_module()
# __convert-sa-to-ma-end__


# __write-custom-sa-rlmodule-torch-begin__
from typing import Any, Dict

from ray.rllib.core import Columns
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule


class DiscreteBCTorchModule(TorchRLModule):
    def setup(self):
        input_dim = self.observation_space.shape[0]
        hidden_dim = self.model_config["fcnet_hiddens"][0]
        output_dim = self.action_space.n

        self._pi_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def _forward_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            return self._forward_train(batch)

    def _forward_exploration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            return self._forward_train(batch)

    def _forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        action_logits = self._pi_head(batch[Columns.OBS])
        return {"action_dist": torch.distributions.Categorical(logits=action_logits)}


# __write-custom-sa-rlmodule-torch-end__


# __extend-spec-checking-single-level-begin__
class DiscreteBCTorchModule(TorchRLModule):
    ...

    @override(TorchRLModule)
    def input_specs_exploration(self) -> SpecType:
        # Enforce that input nested dict to exploration method has a key "obs"
        return ["obs"]

    @override(TorchRLModule)
    def output_specs_exploration(self) -> SpecType:
        # Enforce that output nested dict from exploration method has a key
        # "action_dist"
        return ["action_dist"]


# __extend-spec-checking-single-level-end__


# __extend-spec-checking-nested-begin__
class DiscreteBCTorchModule(TorchRLModule):
    ...

    @override(TorchRLModule)
    def input_specs_exploration(self) -> SpecType:
        # Enforce that input nested dict to exploration method has a key "obs"
        # and within that key, it has a key "global" and "local". There should
        # also be a key "action_mask"
        return [("obs", "global"), ("obs", "local"), "action_mask"]


# __extend-spec-checking-nested-end__


# __extend-spec-checking-torch-specs-begin__
class DiscreteBCTorchModule(TorchRLModule):
    ...

    @override(TorchRLModule)
    def input_specs_exploration(self) -> SpecType:
        # Enforce that input nested dict to exploration method has a key "obs"
        # and its value is a torch.Tensor with shape (b, h) where b is the
        # batch size (determined at run-time) and h is the hidden size
        # (fixed at 10).
        return {"obs": TensorSpec("b, h", h=10, framework="torch")}


# __extend-spec-checking-torch-specs-end__


# __extend-spec-checking-type-specs-begin__
class DiscreteBCTorchModule(TorchRLModule):
    ...

    @override(TorchRLModule)
    def output_specs_exploration(self) -> SpecType:
        # Enforce that output nested dict from exploration method has a key
        # "action_dist" and its value is a torch.distribution.Categorical
        return {"action_dist": torch.distribution.Categorical}


# __extend-spec-checking-type-specs-end__


# __write-custom-multirlmodule-shared-enc-begin__
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule

import torch
import torch.nn as nn


class BCTorchRLModuleUsingSharedGlobalEncoder(TorchRLModule):
    """An RLModule with a shared encoder between agents for global observation."""

    def __init__(
        self,
        *,
        observation_space,
        action_space,
        model_config,
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config=model_config,
            **kwargs,
        )

        self._encoder = self.model_config["encoder"]
        self._pi_head = nn.Sequential(
            nn.Linear(
                self.model_config["hidden_dim"] + self.model_config["local_dim"],
                self.model_config["hidden_dim"],
            ),
            nn.ReLU(),
            nn.Linear(self.model_config["hidden_dim"], self.action_space.n),
        )

    def _forward_inference(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            return self._common_forward(batch)

    def _forward_exploration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            return self._common_forward(batch)

    def _forward_train(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self._common_forward(batch)

    def _common_forward(self, batch):
        obs = batch["obs"]
        global_enc = self._encoder(obs["global"])
        policy_in = torch.cat([global_enc, obs["local"]], dim=-1)
        logits = self._pi_head(policy_in)

        return {Columns.ACTION_DIST_INPUTS: logits}


class BCTorchMultiAgentModuleWithSharedEncoder(MultiRLModule):
    def setup(self):
        global_dim = self.rl_module_specs.observation_space["global"].shape[0]
        hidden_dim = self.rl_module_specs.model_config["fcnet_hiddens"][0]
        shared_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        rl_modules = {}
        for module_id, module_spec in self.rl_module_specs.items():
            rl_modules[module_id] = BCTorchRLModuleWithSharedGlobalEncoder(
                config=module_specs[module_id].get_rl_module_config(),
                encoder=shared_encoder,
                local_dim=module_spec.observation_space["local"].shape[0],
                hidden_dim=hidden_dim,
                action_dim=module_spec.action_space.n,
            )

        self._rl_modules = rl_modules


# __write-custom-multirlmodule-shared-enc-end__


# __pass-custom-multirlmodule-shared-enc-begin__
import gymnasium as gym
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

spec = MultiRLModuleSpec(
    multi_rl_module_class=BCTorchMultiAgentModuleWithSharedEncoder,
    rl_module_specs={
        "local_2d": RLModuleSpec(
            observation_space=gym.spaces.Dict(
                {
                    "global": gym.spaces.Box(low=-1, high=1, shape=(2,)),
                    "local": gym.spaces.Box(low=-1, high=1, shape=(2,)),
                }
            ),
            action_space=gym.spaces.Discrete(2),
            model_config={"fcnet_hiddens": [64]},
        ),
        "local_5d": RLModuleSpec(
            observation_space=gym.spaces.Dict(
                {
                    "global": gym.spaces.Box(low=-1, high=1, shape=(2,)),
                    "local": gym.spaces.Box(low=-1, high=1, shape=(5,)),
                }
            ),
            action_space=gym.spaces.Discrete(5),
            model_config={"fcnet_hiddens": [64]},
        ),
    },
)

module = spec.build()
# __pass-custom-multirlmodule-shared-enc-end__


# __checkpointing-begin__
import gymnasium as gym
import shutil
import tempfile
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec

config = (
    PPOConfig()
    # Enable the new API stack (RLModule and Learner APIs).
    .api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    ).environment("CartPole-v1")
)
env = gym.make("CartPole-v1")
# Create an RL Module that we would like to checkpoint
module_spec = RLModuleSpec(
    module_class=PPOTorchRLModule,
    observation_space=env.observation_space,
    action_space=env.action_space,
    # If we want to use this externally created module in the algorithm,
    # we need to provide the same config as the algorithm. Any changes to
    # the defaults can be given via the right side of the `|` operator.
    model_config_dict=config.model_config | {"fcnet_hiddens": [32]},
    catalog_class=PPOCatalog,
)
module = module_spec.build()

# Create the checkpoint.
module_ckpt_path = tempfile.mkdtemp()
module.save_to_path(module_ckpt_path)

# Create a new RLModule from the checkpoint.
loaded_module = RLModule.from_checkpoint(module_ckpt_path)

# Create a new Algorithm (with the changed module config: 32 units instead of the
# default 256; otherwise loading the state of `module` will fail due to a shape
# mismatch).
config.rl_module(model_config_dict=config.model_config | {"fcnet_hiddens": [32]})
algo = config.build()
# Now load the saved RLModule state (from the above `module.save_to_path()`) into the
# Algorithm's RLModule(s). Note that all RLModules within the algo get updated, the ones
# in the Learner workers and the ones in the EnvRunners.
algo.restore_from_path(
    module_ckpt_path,  # <- NOT an Algorithm checkpoint, but single-agent RLModule one.
    # We have to provide the exact component-path to the (single) RLModule
    # within the algorithm, which is:
    component="learner_group/learner/rl_module/default_policy",
)

# __checkpointing-end__
algo.stop()
shutil.rmtree(module_ckpt_path)