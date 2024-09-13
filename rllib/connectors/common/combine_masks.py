from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.spaces.space_utils import batch as batch_fn
from ray.rllib.utils.typing import EpisodeType
from ray.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class CombineMasks(ConnectorV2):
    """Combines all masks registered in shared data under the `_masks` set.

    Note that by convention, all RLlib masks should use a value of 0.0 (or False) for
    invalid/masked positions and a value of 1.0 (or True) for valid positions.
    """
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ):
        # TODO (sven): Make mask registering an official thing in a user-friendly
        #  ConnectorV2 API.
        if shared_data is None or "_masks" not in shared_data:
            return batch

        mask_names = shared_data["_masks"]
        assert isinstance(mask_names, set)

        for module_id, module_batch in batch.items():
            total_mask = None
            for mask_name in mask_names:
                if mask_name in module_batch:
                    if total_mask is None:
                        total_mask = module_batch[mask_name].astype(np.bool_)
                    else:
                        total_mask = np.logical_and(total_mask, module_batch[mask_name])
                    #TODO (sven): Activate deletion of old masks.
                    #del module_batch[mask_name]

            if total_mask is not None:
                module_batch[Columns.LOSS_MASK] = total_mask

        return batch
