import torch
from typing import Any, Dict, TYPE_CHECKING

from ray.rllib.connectors.learner import ComputeReturnsToGo
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.core.testing.testing_learner import BaseTestingLearner
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID, TensorType

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig


class VPGTorchLearner(TorchLearner, BaseTestingLearner):
    @override(TorchLearner)
    def build(self) -> None:
        super().build()

        # Prepend the returns-to-go connector piece to have that information
        # available in the train batch.
        if self.config.add_default_connectors_to_learner_pipeline:
            self._learner_connector.prepend(ComputeReturnsToGo())

    @override(TorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: "AlgorithmConfig",
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        BaseTestingLearner.compute_loss_for_module(
            self,
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )
        rl_module = self.module[module_id]
        action_dist_inputs = fwd_out[Columns.ACTION_DIST_INPUTS]
        action_dist_class = rl_module.get_train_action_dist_cls()
        action_dist = action_dist_class.from_logits(action_dist_inputs)

        # Compute log probabilities of the actions taken
        log_probs = action_dist.logp(batch[Columns.ACTIONS])

        # Compute the policy gradient loss
        # Since we're not using a baseline, we use returns directly
        loss = -torch.mean(log_probs * batch[Columns.RETURNS_TO_GO])

        return loss
