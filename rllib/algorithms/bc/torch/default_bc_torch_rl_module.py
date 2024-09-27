from ray.rllib.algorithms.bc.default_bc_rl_module import DefaultBCRLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule


class DefaultBCTorchRLModule(TorchRLModule, DefaultBCRLModule):
    pass
