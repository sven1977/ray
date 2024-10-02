from typing import Type

import numpy as np

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.learner.learner import Learner
from ray.rllib.core.rl_module.multi_rl_module import (
    MultiRLModule,
    MultiRLModuleSpec,
)
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import RLModuleSpecType


class BaseTestingAlgorithmConfig(AlgorithmConfig):
    # A test setting to activate metrics on mean weights.
    report_mean_weights: bool = True

    @override(AlgorithmConfig)
    def get_default_learner_class(self) -> Type["Learner"]:
        if self.framework_str == "torch":
            from ray.rllib.examples.learners.classes.vpg_learner import VPGTorchLearner

            return VPGTorchLearner
        else:
            raise ValueError(f"Unsupported framework: {self.framework_str}")

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self) -> "RLModuleSpecType":
        if self.framework_str == "torch":
            from ray.rllib.examples.rl_modules.classes.vpg_rlm import VPGTorchRLModule

            cls = VPGTorchRLModule
        else:
            raise ValueError(f"Unsupported framework: {self.framework_str}")

        spec = RLModuleSpec(
            module_class=cls,
            model_config_dict={"fcnet_hiddens": [32]},
        )

        if self.is_multi_agent():
            # TODO (sven): Make this more multi-agent for example with policy ids
            #  "p0" and "p1".
            return MultiRLModuleSpec(
                multi_rl_module_class=MultiRLModule,
                module_specs={DEFAULT_MODULE_ID: spec},
            )
        else:
            return spec


class BaseTestingLearner(Learner):
    @override(Learner)
    def after_gradient_based_update(self, *, timesteps):
        # This is to check if in the multi-gpu case, the weights across workers are
        # the same. It is really only needed during testing.
        if self.config.report_mean_weights:
            for module_id in self.module.keys():
                parameters = convert_to_numpy(
                    self.get_parameters(self.module[module_id])
                )
                mean_ws = np.mean([w.mean() for w in parameters])
                self.metrics.log_value((module_id, "mean_weight"), mean_ws, window=1)
