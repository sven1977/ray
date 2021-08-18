#TODO
import argparse
import os

import numpy as np

import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.examples.env.repeat_after_me_env import RepeatAfterMeEnv
from ray.rllib.models.v3.model_with_value_function import \
    RNNModel, RNNModelWithValueFunction
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print

tf1, tf, tfv = try_import_tf()
SUPPORTED_ENVS = ["CartPole-v0", "Pendulum-v0"]


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    # example-specific args
    parser.add_argument(
        "--env", choices=SUPPORTED_ENVS, default=SUPPORTED_ENVS[0])

    # general args
    parser.add_argument(
        "--run", default="PPO", help="The RLlib-registered algorithm to use.")
    parser.add_argument("--num-cpus", type=int, default=3)
    parser.add_argument(
        "--framework",
        choices=["tf", "tf2", "tfe", "torch"],
        default="tf",
        help="The DL framework specifier.")
    parser.add_argument(
        "--stop-iters",
        type=int,
        default=200,
        help="Number of iterations to train.")
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=500000,
        help="Number of timesteps to train.")
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=80.0,
        help="Reward at which we stop training.")
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test: --stop-reward must "
        "be achieved within --stop-timesteps AND --stop-iters.")
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Run without Tune using a manual train loop instead. Here,"
        "there is no TensorBoard support.")
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.")

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


if __name__ == "__main__":
    args = get_cli_args()

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    # ModelV3 config w/ shared value function.
    shared_vf = {
        "policy_model": {
            "shared": {
                "fcnet_hiddens": [64, 64],
            },
            "policy": {
                "input_source": "shared",
                "fcnet_hiddens": [],
                "output_layer_size": "action_space",
            },
            "value": {
                "input_source": "shared",
                "fcnet_hiddens": [],
                "output_layer_size": 1,
            },
        },
    }

    separate_vf = {
        "policy_model": {
            "fcnet_hiddens": [128, 128],
            "output_layer_size": "action_space",
        },
        "value_model": {
            "fcnet_hiddens": [64, 64],
            "output_layer_size": 1,
        },
    }

    separate_vf_w_lstm_on_both = {
        "policy_model": {
            "fcnet_hiddens": [64, 64],
            "output_layer_size": "action_space",
            "use_lstm": True,
            "lstm_cell_size": 128,
        },
        "value_model": {
            "fcnet_hiddens": [32, 32],
            "output_layer_size": 1,
            "use_lstm": True,
            "lstm_cell_size": 64,
        },
    }

    separate_vf_w_lstm_only_on_policy = {
        "policy_model": {
            "fcnet_hiddens": [32, 32],
            "output_layer_size": "action_space",
            "use_lstm": True,
            "lstm_cell_size": 128,
        },
        "value_model": {
            "fcnet_hiddens": [64, 64],
            "output_layer_size": 1,
        },
    }

    # Using a full custom model (including the value calculations).
    # The given Model must abide by the PPO model API, which is:
    # 1) __call__() == forward_policy_and_value()
    # 2) forward_policy
    # 3) forward_value
    custom_model_incl_vf = {
        "policy_model": {
            "custom_model": RNNModelWithValueFunction,
            "custom_model_config": {
                "hiddens_size": 16,
                "cell_size": 10,
            },
        },
    }

    # Using a custom shared branch (w/ RNN) underneath default
    # policy- and value heads.
    custom_shared_branch = {
        "policy_model": {
            "shared": {
                "custom_model": RNNModel,
                "custom_model_config": {
                    "hiddens_size": 10,
                    "cell_size": 11,
                },
            },
            "policy_head": {

            },
            "value_head": {

            },
        },
    }

    local_vars = locals()

    # main part: RLlib config with AttentionNet model
    config = {
        "env": args.env,
        "gamma": 0.99,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", 0)),
        "num_envs_per_worker": 20,
        "entropy_coeff": 0.001,
        "num_sgd_iter": 10,
        "vf_loss_coeff": 1e-5,
        # Use the new `_models` config key indicating we would like to
        # use the ModelV3-builder API.
        "_models": local_vars[args.modelv3_setup],
        "framework": args.framework,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # training loop
    if args.no_tune:
        # manual training loop using PPO and manually keeping track of state
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(config)
        trainer = ppo.PPOTrainer(config=ppo_config, env=args.env)
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = trainer.train()
            print(pretty_print(result))
            # stop training if the target train steps or reward are reached
            if result["timesteps_total"] >= args.stop_timesteps or \
                    result["episode_reward_mean"] >= args.stop_reward:
                break

        # run manual test loop (only for RepeatAfterMe env)
        if args.env == "RepeatAfterMeEnv":
            print("Finished training. Running manual test/inference loop.")
            # prepare env
            env = RepeatAfterMeEnv(config["env_config"])
            obs = env.reset()
            done = False
            total_reward = 0
            # start with all zeros as state
            num_transformers = config["model"][
                "attention_num_transformer_units"]
            init_state = state = [
                np.zeros([100, 32], np.float32)
                for _ in range(num_transformers)
            ]
            # run one iteration until done
            print(f"RepeatAfterMeEnv with {config['env_config']}")
            while not done:
                action, state_out, _ = trainer.compute_single_action(
                    obs, state)
                next_obs, reward, done, _ = env.step(action)
                print(f"Obs: {obs}, Action: {action}, Reward: {reward}")
                obs = next_obs
                total_reward += reward
                state = [
                    np.concatenate([state[i], [state_out[i]]], axis=0)[1:]
                    for i in range(num_transformers)
                ]
            print(f"Total reward in test episode: {total_reward}")

    else:
        # run with Tune for auto env and trainer creation and TensorBoard
        results = tune.run(args.run, config=config, stop=stop, verbose=2)

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()
