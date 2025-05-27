import abc
from typing import Any, Dict

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.torch.primitives.encoder import build_encoder
from ray.rllib.core.rl_module.torch.primitives.mlp_head import build_mlp_head
from ray.rllib.utils.annotations import override
from ray.util.annotations import DeveloperAPI


@DeveloperAPI
class DefaultBCTorchRLModule(TorchRLModule, abc.ABC):
    """The default TorchRLModule used for BC, if no custom RLModule is provided.

    Builds an encoder net based on the observation space.
    Builds a pi head based on the action space.

    obs-batch -> encoder -> 1D latent -> pi-head -> action dist. parameters
    """

    @override(RLModule)
    def setup(self):
        super().setup()

        # Build the encoder.
        self._encoder, output_dims = build_encoder(
            self.observation_space,
            self.model_config,
        )
        # Build the policy-head.
        self._pi_head = build_mlp_head(
            input_dim=output_dims[0],
            model_config=self.model_config,
            action_dist_class=self.get_inference_action_dist_cls(),
            action_space=self.action_space,
        )

    @override(TorchRLModule)
    def _forward(self, batch: Dict, **kwargs) -> Dict[str, Any]:
        """Generic BC forward pass (for all phases of training/evaluation)."""

        # Encoder embeddings.
        encoder_outs = self._encoder(batch)
        # Pi-output.
        logits = self._pi_head(encoder_outs)

        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }
