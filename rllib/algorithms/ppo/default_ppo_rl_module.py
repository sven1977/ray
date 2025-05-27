import abc
from typing import List

#from ray.rllib.core.models.configs import RecurrentEncoderConfig
from ray.rllib.core.rl_module import ACTOR, CRITIC
from ray.rllib.core.rl_module.apis import InferenceOnlyAPI, ValueFunctionAPI
#from ray.rllib.core.rl_module.primitives import RecurrentEncoderConfig
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.primitives import build_encoder, build_mlp_head
from ray.rllib.utils.annotations import (
    override,
    OverrideToImplementCustomLogic_CallToSuperRecommended,
)
from ray.util.annotations import DeveloperAPI


@DeveloperAPI
class DefaultPPORLModule(RLModule, InferenceOnlyAPI, ValueFunctionAPI, abc.ABC):
    """Default RLModule used by PPO, if user does not specify a custom RLModule.

    Users who want to train their RLModules with PPO may implement any RLModule
    (or TorchRLModule) subclass as long as the custom class also implements the
    `ValueFunctionAPI` (see ray.rllib.core.rl_module.apis.value_function_api.py)
    """

    @override(RLModule)
    def setup(self):
        super().setup()

        # If we have a stateful model (LSTM), states for the critic need to be collected
        # during sampling and `inference-only` needs to be `False`.
        if self.model_config["use_lstm"]:
            self.inference_only = False

        self._encoder = self._pi_encoder = self._vf_encoder = None
        if self.model_config["vf_share_layers"]:
            self._encoder, latent_dims = build_encoder(
                self.observation_space,
                self.model_config,
            )
        else:
            self._pi_encoder, latent_dims = build_encoder(
                self.observation_space,
                self.model_config,
            )
            self._vf_encoder, latent_dims = build_encoder(
                self.observation_space,
                self.model_config,
            )

        # __sphinx_doc_begin__
        #is_stateful = isinstance(
        #    self.catalog.actor_critic_encoder_config.base_encoder_config,
        #    RecurrentEncoderConfig,
        #)

        ## If this is an `inference_only` Module, we'll have to pass this information
        ## to the encoder config as well.
        #if self.inference_only and self.framework == "torch":
        #    self.catalog.actor_critic_encoder_config.inference_only = True

        self._pi_head = build_mlp_head(
            latent_dims[0],
            self.model_config,
            action_dist_class=self.get_inference_action_dist_cls(),
            action_space=self.action_space,
        )
        self._vf_head = build_mlp_head(
            latent_dims[0],
            self.model_config,
            output_dim=1,
        )
        # __sphinx_doc_end__

    @override(RLModule)
    def get_initial_state(self) -> dict:
        if self._encoder:
            return self._encoder.get_initial_state()

        state = {}
        pi_encoder_state = self._pi_encoder.get_initial_state()
        if pi_encoder_state:
            state[ACTOR] = pi_encoder_state
        vf_encoder_state = self._vf_encoder.get_initial_state()
        if vf_encoder_state:
            state[CRITIC] = vf_encoder_state

        return state

    @OverrideToImplementCustomLogic_CallToSuperRecommended
    @override(InferenceOnlyAPI)
    def get_non_inference_attributes(self) -> List[str]:
        """Return attributes, which are NOT inference-only (only used for training)."""
        return ["vf"] + (
            []
            if self.model_config.get("vf_share_layers")
            else ["encoder.critic_encoder"]
        )
