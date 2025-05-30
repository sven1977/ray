from typing import Any, Dict, Optional

from ray.rllib.algorithms.ppo.default_ppo_rl_module import DefaultPPORLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module import ACTOR, CRITIC
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
from ray.util.annotations import DeveloperAPI

torch, nn = try_import_torch()


@DeveloperAPI
class DefaultPPOTorchRLModule(TorchRLModule, DefaultPPORLModule):
    @override(RLModule)
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Default forward pass (used for inference and exploration)."""
        output = {}
        # Encoder forward pass(es).
        if self._encoder:
            pi_encoder_outs = self._encoder(batch)
        else:
            pi_encoder_outs = self._separate_encoder_forward(batch, vf=False)
        # Stateful encoder?
        if Columns.STATE_OUT in pi_encoder_outs:
            output[Columns.STATE_OUT] = pi_encoder_outs[Columns.STATE_OUT]
        # Pi head.
        output[Columns.ACTION_DIST_INPUTS] = self._pi_head(
            pi_encoder_outs[Columns.EMBEDDINGS]
        )
        return output

    @override(RLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Train forward pass (keep embeddings for possible shared value func. call)."""
        output = {}
        if self._encoder:
            pi_encoder_outs = vf_encoder_outs = self._encoder(batch)
        else:
            pi_encoder_outs = self._separate_encoder_forward(batch, vf=False)
            vf_encoder_outs = self._separate_encoder_forward(batch, vf=True)
        output[Columns.EMBEDDINGS] = vf_encoder_outs[Columns.EMBEDDINGS]
        # if Columns.STATE_OUT in encoder_outs:
        #    output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]
        output[Columns.ACTION_DIST_INPUTS] = self._pi_head(
            pi_encoder_outs[Columns.EMBEDDINGS]
        )
        return output

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        if embeddings is None:
            # Separate vf-encoder.
            if self._vf_encoder:
                encoder_outs = self._separate_encoder_forward(batch, vf=True)
            # Shared encoder.
            else:
                encoder_outs = self._encoder(batch)
            embeddings = encoder_outs[Columns.EMBEDDINGS]

        # Value head.
        vf_out = self._vf_head(embeddings)
        # Squeeze out last dimension (single node value head).
        return vf_out.squeeze(-1)

    def _separate_encoder_forward(self, batch, *, vf=False):
        batch_ = batch
        if self.is_stateful():
            # The recurrent encoders expect a `(state_in, h)`  key in the
            # input dict while the key returned is `(state_in, critic, h)`.
            batch_ = batch.copy()
            key = CRITIC if vf else ACTOR
            if (
                isinstance(batch[Columns.STATE_IN], dict)
                and key in batch[Columns.STATE_IN]
            ):
                batch_[Columns.STATE_IN] = batch[Columns.STATE_IN][key]

        if vf:
            return self._vf_encoder(batch_)
        else:
            return self._pi_encoder(batch_)
