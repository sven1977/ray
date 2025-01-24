from typing import Any, Dict, Optional

from ray.rllib.algorithms.maxent_irl.maxent_irl import MaxEntIRLConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()


class MaxEntIRLTorchLearner(TorchLearner):
    """Implements torch-specific MaxEntIRL loss on top of MaxEntIRLLearner.

    This class implements the MaxEntIRL loss under `self.compute_loss_for_module()`.
    """

    def compute_loss_for_module(
        self,
        *,
        module_id: str,
        config: Optional[MaxEntIRLConfig] = None,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType]
    ) -> TensorType:
        # module = self.module[module_id].unwrapped()

        predicted_rewards = fwd_out[Columns.REWARDS]
        predicted_observed_rewards, predicted_sampled_rewards = torch.split(predicted_rewards)

        # Compute observed trajectory rewards
        observed_trajectory_rewards = predicted_observed_rewards.sum(dim=1)

        # Approximate log(Z) using sampled trajectories
        sampled_trajectory_rewards = predicted_sampled_rewards.detach().sum(dim=1)

        # Compute log(Z).
        log_Z = torch.logsumexp(sampled_trajectory_rewards, dim=0)

        # Compute MaxEnt IRL loss
        total_loss = -torch.mean(observed_trajectory_rewards - log_Z)

        # Return the total loss.
        return total_loss
