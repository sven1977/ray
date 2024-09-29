import abc
from typing import Any, Dict

from ray.rllib.core.columns import Columns
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.util.annotations import DeveloperAPI


@DeveloperAPI(stability="alpha")
class DefaultBCTorchRLModule(TorchRLModule, abc.ABC):
    """The default RLModule used (framework=torch), if no custom RLModule is provided.

    Builds an encoder net based on the observation space.
    Builds a pi head based on the action space.

    Passes observations from the input batch through the encoder, then the pi head to
    compute action logits.
    """

    @override(RLModule)
    def setup(self):
        # Build model components (encoder and pi head) from catalog.
        self._encoder = self.catalog.build_encoder(framework=self.framework)
        self._pi_head = self.catalog.build_pi_head(framework=self.framework)

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict, **kwargs) -> Dict[str, Any]:
        """BC forward pass for action inference."""
        output = {}
        # State encodings.
        encoder_outs = self._encoder(batch)
        # Actions.
        action_logits = self._pi_head(encoder_outs[ENCODER_OUT])
        output[Columns.ACTION_DIST_INPUTS] = action_logits

        return output

    @override(RLModule)
    def _forward_exploration(self, batch: Dict, **kwargs) -> Dict[str, Any]:
        """BC forward pass for exploratory behavior."""
        return self._forward_inference(batch)

    @override(RLModule)
    def _forward_train(self, batch: Dict, **kwargs) -> Dict[str, Any]:
        """BC forward pass for training."""
        return self._forward_inference(batch)

    @override(RLModule)
    def output_specs_train(self) -> SpecType:
        return [Columns.ACTION_DIST_INPUTS]
