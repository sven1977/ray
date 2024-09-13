"""Example of implementing and configuring a custom (torch) transformer RLModule.

This example:
    - demonstrates how you can subclass the TorchRLModule base class and set up your
    own transformer/attention net architecture by overriding the `setup()` method.
    - shows how to override the 3 forward methods: `_forward_inference()`,
    `_forward_exploration()`, and `forward_train()` to implement your own custom forward
    logic(s). You will also learn, when each of these 3 methods is called by RLlib or
    the users of your RLModule.
    - shows how you then configure an RLlib Algorithm such that it uses your custom
    RLModule (instead of a default RLModule).
    - because you are using a transformer architecture, one also requires custom
    env-to-module and Learner connector pieces in order to add the last n observations
    to the forward batches (instead of just the most recent one). The necessary
    ConnectorV2 classes are imported in this script and then added to the respective
    connector pipelines through the algorithm config.


For more details on the used transformer architecture, see this file here:
ray.rllib.examples.rl_modules.classes.custom_transformer_rl_module.py

The network is used in a minigrid memory/attention-requiring experiment.


How to run this script
----------------------
`python [script file name].py --enable-new-api-stack`

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`


Results to expect
-----------------
You should see the following output (during the experiment) in your console:

"""
import gymnasium as gym

from ray.rllib.connectors.env_to_module.add_transformer_input_to_batch import (
    AddTransformerInputToBatchEnvToModule
)
from ray.rllib.connectors.learner.add_transformer_input_to_batch import (
    AddTransformerInputToBatchLearner
)
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.rl_modules.classes.transformer_simple_rlm import (
    TransformerSimple
)
from ray.rllib.examples.envs.env_rendering_and_recording import EnvRenderCallback
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env

parser = add_rllib_example_script_args(default_reward=450.0, default_iters=10000, default_timesteps=10000000)
parser.set_defaults(
    enable_new_api_stack=True,
)


if __name__ == "__main__":
    args = parser.parse_args()

    from ray.rllib.examples.envs.minigrid_env import ImgDirectionWrapper

    register_env(
        "env",
        lambda cfg: ImgDirectionWrapper(gym.make("MiniGrid-MemoryS11-v0", render_mode="rgb_array")),
    )

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment("env")
        .env_runners(
            env_to_module_connector=lambda env: AddTransformerInputToBatchEnvToModule(),
        )
        .training(
        #    learner_connector=lambda in_o, in_a: AddTransformerInputToBatchLearner(),
            lr=[[0, 0.00005], [200000, 0.0001]],
            sgd_minibatch_size=256,
            num_sgd_iter=6,
            vf_loss_coeff=1.0,
            entropy_coeff=0.005,
            #grad_clip=1.0,
        )
        .rl_module(
            # Plug-in our custom RLModule class.
            rl_module_spec=RLModuleSpec(
                module_class=TransformerSimple,
            ),
            # Feel free to specify your own `model_config_dict` settings below.
            # The `model_config_dict` defined here will be available inside your
            # custom RLModule class through the `self.config.model_config_dict`
            # property.
            model_config_dict={
                # The maximum number of timesteps to feed into the attention net
                # (this is for both inference and training batches).
                "max_seq_len": 40,
                # The number of transformer units within the model.
                "attention_num_transformer_units": 1,
                # The input and output size of each transformer unit.
                "attention_dim": 128,
                # The number of attention heads within a multi-head unit.
                "attention_num_heads": 1,
                # The number of nodes in the position-wise MLP layers
                # (2 layers with ReLU in between) following the self-attention
                # sub-layer within a transformer unit.
                "attention_position_wise_mlp_dim": 256,
            },
        )
        .evaluation(
            evaluation_num_env_runners=1,
            evaluation_interval=10,
            evaluation_duration=1,
            evaluation_parallel_to_training=True,
            evaluation_config={
                "explore": True,
                "callbacks": EnvRenderCallback,
            },
        )
    )

    run_rllib_example_script_experiment(base_config, args)
