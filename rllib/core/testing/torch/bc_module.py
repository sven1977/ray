from typing import Any, Dict

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class BCTorchRLModuleWithSharedGlobalEncoder(TorchRLModule):
    """An example of an RLModule that uses an encoder shared with other things.

    For example, we could consider a multi-agent case where for inference each agent
    needs to know the global state of the environment, as well as the local state of
    itself. For better representation learning we would like to share the encoder
    across all the modules. So this module simply accepts the encoder object as its
    input argument and uses it to encode the global state. The local state is passed
    through as is. The policy head is then a simple MLP that takes the concatenation of
    the global and local state as input and outputs the action logits.

    """

    def setup(self):
        super().setup()

        feature_dim = self.model_config["feature_dim"]
        hidden_dim = self.model_config["hidden_dim"]

        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_space.n),
        )

    @override(RLModule)
    def _forward_inference(self, batch):
        with torch.no_grad():
            return self._common_forward(batch)

    @override(RLModule)
    def _forward_exploration(self, batch):
        with torch.no_grad():
            return self._common_forward(batch)

    @override(RLModule)
    def _forward_train(self, batch):
        return self._common_forward(batch)

    def _common_forward(self, batch):
        action_logits = self.policy_head(batch["encoder_features"])
        return {Columns.ACTION_DIST_INPUTS: action_logits}


class BCTorchMultiAgentModuleWithSharedEncoder(MultiRLModule):
    #def setup(self):
    #    super().setup()

    #    module_specs = self.config.modules
    #    module_spec = next(iter(module_specs.values()))
    #    global_dim = module_spec.observation_space["global"].shape[0]
    #    hidden_dim = module_spec.model_config_dict["fcnet_hiddens"][0]
    #    shared_encoder = nn.Sequential(
    #        nn.Linear(global_dim, hidden_dim),
    #        nn.ReLU(),
    #        nn.Linear(hidden_dim, hidden_dim),
    #    )

        #rl_modules = {}
        #for module_id, module_spec in module_specs.items():
        #    rl_modules[module_id] = module_spec.module_class(
        #        config=self.config.modules[module_id].get_rl_module_config(),
        #        encoder=shared_encoder,
        #        local_dim=module_spec.observation_space["local"].shape[0],
        #        hidden_dim=hidden_dim,
        #        action_dim=module_spec.action_space.n,
        #    )

        #self._rl_modules = rl_modules

    def _forward_inference(self, batch, **kwargs):

