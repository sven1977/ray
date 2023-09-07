import argparse
from collections import deque
from functools import partial
import os
import tempfile
import zipfile

import flappy_bird_gymnasium  # noqa
import gymnasium as gym
import numpy as np
from supersuit.generic_wrappers import resize_v1
import tensorflow as tf
import tree  # pip install dm_tree
import wandb

import ray
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray.rllib.algorithms.dreamerv3.utils.env_runner import NormalizedImageEnv
from ray.rllib.algorithms.simple_q import SimpleQConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wandb-key",
        type=str,
        default=None,  # Or set the env variable $WANDB_KEY to your API key.
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="test",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="test-run",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--num-eval-eps",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--restore-from-iteration",
        type=int,
        default=None,
    )

    return parser.parse_args()


def reinforcement_learning_experiment(
    *,
    config: AlgorithmConfig,
    wandb_key: str,
    wandb_project: str,
    wandb_run_name: str,
    eval_freq: int,
    eval_eps: int,
    checkpoint_freq: int = 200,
):
    # Login to WandB (this step can be optional depending on how you manage your WandB
    # API key)
    wandb.login(key=wandb_key)
    # Initialize WandB.
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        resume=True,
        config=config.to_dict(),
    )

    # Create the placement group.
    # placement_group_factory = config.algo_class.default_resource_request(config)
    # pg = ray.util.placement_group(
    #    strategy=placement_group_factory.strategy,
    #    bundles=placement_group_factory.bundles,
    # )
    # Wait until placement group is created.
    # ray.get(pg.ready(), timeout=300)

    # Create or load Trainer.
    if args.restore_from_iteration:
        artifact = wandb.use_artifact(
            f"{args.wandb_run_name}_{args.restore_from_iteration:08}:latest")
        artifact.download("artifacts/")
        # Extract the checkpoint.
        with zipfile.ZipFile(
            f"artifacts/checkpoint_{args.restore_from_iteration:08}.zip", "r") as zipf:
            zipf.extractall("artifacts/")
        algo = Algorithm.from_checkpoint("artifacts")
        print(f"Restored Algo from checkpoint (iteration={algo.training_iteration}).")
    else:
        algo = config.build()
        print("Created new Algo from scratch.")

    # Initialize evaluation tracking
    highest_avg_return = -float('inf')
    ema = None  # Exponential Moving Average
    average_returns = deque(maxlen=100)

    # Create an eval gymnasium environment
    env = NormalizedImageEnv(
        resize_v1(
            gym.make(
                config.env,
                **config.env_config,
                render_mode="rgb_array",
            ),
            x_size=64,
            y_size=64,
        )
    )
    module = algo.workers.local_worker().module

    # Test quick forward pass through module.
    states = module.get_initial_state()
    is_first = np.ones((1,))
    observation, _ = env.reset()
    batch = {
        "state_in": tree.map_structure(
            lambda s: tf.convert_to_tensor(s), states
        ),
        "obs": tf.convert_to_tensor(np.expand_dims(observation, axis=0)),
        "is_first": tf.convert_to_tensor(is_first),
    }
    outs = algo.workers.local_worker().module.forward_inference(batch)

    # Main loop
    while True:
        print(f"Training iteration {algo.training_iteration}")

        # Training phase.
        results = algo.train()
        results.pop("config")
        wandb.log(results, step=algo.training_iteration)
        print(f"\ttraining return={results['sampler_results']['episode_reward_mean']}")

        videos = []

        # Evaluation phase.
        avg_return = None
        if algo.training_iteration % eval_freq == 0:
            episode_rewards = []

            for episode in range(eval_eps):
                video = []

                states = module.get_initial_state()
                is_first = np.ones((1,))
                observation, _ = env.reset()

                video.append(env.render())

                terminated = truncated = False
                episode_reward = 0.0
                while not (terminated or truncated):
                    batch = {
                        "state_in": tree.map_structure(
                            lambda s: tf.convert_to_tensor(s), states
                        ),
                        "obs": tf.convert_to_tensor(
                            np.expand_dims(observation, axis=0)),
                        "is_first": tf.convert_to_tensor(is_first),
                    }
                    outs = algo.workers.local_worker().module.forward_inference(batch)

                    action = action_env = outs["actions"].numpy()
                    if isinstance(env.action_space, gym.spaces.Discrete):
                        action_env = np.argmax(action, axis=-1)
                    states = tree.map_structure(lambda s: s.numpy(), outs["state_out"])

                    observation, reward, terminated, truncated, _ = env.step(action_env)
                    video.append(env.render())
                    episode_reward += reward

                    is_first[0] = 0.0

                videos.append(video)
                episode_rewards.append(episode_reward)

            avg_return = np.mean(episode_rewards)
            average_returns.append(avg_return)
            print(f"Average eval episode return: {avg_return}")
            wandb.log({"EVAL_average_return": avg_return}, step=algo.training_iteration)

            # Save eval videos (worst and best performance)?
            # Highest over last n iterations.
            if (
                avg_return == max(average_returns)
            ):
                print("\tuploading eval videos ...")
                wandb.log(
                    {
                        "EVAL_videos_worst_and_best": [
                            wandb.Video(
                                np.transpose(np.stack(videos[int(idx)], axis=0),
                                             axes=[0, 3, 1, 2])
                            ) for idx in
                            [np.argmin(episode_rewards), np.argmax(episode_rewards)]
                        ],
                    },
                    step=algo.training_iteration,
                )

        # Save a checkpoint?
        if (
            # We have evaluated this iteration AND evaluation results are best ever.
            (
                avg_return is not None
                and highest_avg_return < avg_return
            )
            # `checkpoint-freq` setting mandates to checkpoint.
            or algo.training_iteration % checkpoint_freq == 0
        ):
            print("\tsaving algo ...")
            with tempfile.TemporaryDirectory() as tmpdir:
                algo.save(tmpdir)
                zip_file_name = f"checkpoint_{algo.training_iteration:08}.zip"
                zip_file_full = os.path.join(tmpdir, zip_file_name)
                with zipfile.ZipFile(zip_file_full, "w") as zipf:
                    for root, dirs, files in os.walk(tmpdir):
                        for file in files:
                            if file == zip_file_name:
                                continue
                            zipf.write(os.path.join(root, file))

                # Upload checkpoint to WandB.
                artifact = wandb.Artifact(
                    f"{wandb_run_name}_{algo.training_iteration:08}",
                    type="algo-checkpoint",
                )
                artifact.add_file(zip_file_full)
                wandb.log_artifact(artifact)

        if avg_return is not None and avg_return > highest_avg_return:
            print(f"\t!!new eval highscore!!")
            highest_avg_return = avg_return


if __name__ == "__main__":
    args = parse_args()

    # ray.init(address="auto")

    # Number of GPUs to run on.
    num_gpus = 16
    factor = num_gpus ** 0.5

    # DreamerV3 config and default (1 GPU) learning rates.
    config = DreamerV3Config()
    w = config.world_model_lr
    c = config.critic_lr

    (
        config.environment("FlappyBird-rgb-v0", env_config={"audio_on": False})
            .resources(
            num_learner_workers=0 if num_gpus == 1 else num_gpus,
            num_gpus_per_learner_worker=1 if num_gpus else 0,
            num_cpus_for_local_worker=int(min((num_gpus or 1) * 8, 47)),
        )
            .rollouts(
            # If we use >1 GPU and increase the batch size accordingly, we should also
            # increase the number of envs per worker.
            num_envs_per_worker=8 * (num_gpus or 1),
            remote_worker_envs=True,
        )
            .reporting(
            metrics_num_episodes_for_smoothing=(num_gpus or 1),
            report_images_and_videos=False,
            report_dream_data=False,
            report_individual_batch_item_stats=False,
        )
            # See Appendix A.
            .training(
            model_size="M",
            training_ratio=64,
            batch_size_B=16 * (num_gpus or 1),
            # Use a well established 4-GPU lr scheduling recipe:
            # ~ 1000 training updates with 0.4x[default rates], then over a few hundred
            # steps, increase to 4x[default rates].
            world_model_lr=factor * w,
            # [[0, 0.6 * w], [100000, 0.6 * w], [200000, 8 * w]],
            critic_lr=factor * c,  # [[0, 0.6 * c], [100000, 0.6 * c], [200000, 8 * c]],
            actor_lr=factor * c,  # [[0, 0.6 * c], [100000, 0.6 * c], [200000, 8 * c]],
            world_model_grad_clip_by_global_norm=1000 / factor,
            critic_grad_clip_by_global_norm=100 / factor,
            actor_grad_clip_by_global_norm=100 / factor,
        )
    )

    reinforcement_learning_experiment(
        config=config,
        wandb_key=args.wandb_key or os.environ["WANDB_KEY"],
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        eval_freq=args.eval_freq,
        eval_eps=args.num_eval_eps,
        checkpoint_freq=args.checkpoint_freq,
    )
