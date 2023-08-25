import os
import wandb
import gymnasium as gym
import zipfile
from collections import deque
import numpy as np


from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.dreamerv3.dreamerv3 import DreamerV3Config
from ray.rllib.algorithms.simple_q import SimpleQConfig


def reinforcement_learning_experiment(
    *,
    config: AlgorithmConfig,
    wandb_key: str,
    wandb_project: str,
    wandb_run_name: str,
    eval_freq: int,
    eval_eps: int,
    alpha: float = 0.1,
):
    # Login to WandB (this step can be optional depending on how you manage your WandB
    # API key)
    wandb.login(key=wandb_key)

    # Initialize WandB
    wandb.init(project=wandb_project, name=wandb_run_name, resume="allow", config=config.to_dict())

    # Create or load Trainer.
    try:
        artifact = wandb.use_artifact(f"{wandb_run_name}:latest")
        artifact.download(target_dir="artifacts/")
        algo = Algorithm.from_checkpoint("artifacts/checkpoint")
        print(f"Restored Algo from checkpoint (iteration={algo.training_iteration}).")
    except:
        algo = config.build()
        print("Created new Algo from scratch.")

    # Initialize evaluation tracking
    highest_avg_return = -float('inf')
    #recent_evaluations = deque(maxlen=o)
    ema = None  # Exponential Moving Average

    # Create an eval gymnasium environment
    env = gym.make(config.env, **config.env_config, render_mode="rgb_array")

    # Main loop
    while True:
        # Training phase.
        results = algo.train()
        wandb.log(results, step=algo.training_iteration)

        videos = []

        # Evaluation phase.
        if algo.training_iteration % eval_freq == 0:
            print("Starting evaluation phase ...")
            episode_rewards = []

            for episode in range(eval_eps):
                video = []
                observation, _ = env.reset()
                video.append(env.render())
                terminated = truncated = False
                episode_reward = 0.0

                while not (terminated or truncated):
                    action = algo.compute_single_action(observation, explore=False)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    video.append(env.render())
                    episode_reward += reward

                videos.append(video)
                episode_rewards.append(episode_reward)

            avg_return = np.mean(episode_rewards)
            print(f"Average episode return: {avg_return}")
            wandb.log({"EVAL_average_return": avg_return}, step=algo.training_iteration)

            # Update EMA
            ema = avg_return if ema is None else (1 - alpha) * ema + alpha * avg_return

            # Update the deque (fixed-size list) of recent evaluations
            #recent_evaluations.append(avg_return)
            #avg_of_recent = np.mean(recent_evaluations)

            # You can either use EMA or the average of recent evaluations to
            # decide whether to save the model.
            if ema > highest_avg_return:
                highest_avg_return = ema  # replace `ema` with `avg_of_recent` if desired
                print(f"New highest average return: {highest_avg_return}")

                # Save algo.
                algo.save("checkpoint")
                with zipfile.ZipFile("checkpoint.zip", "w") as myzip:
                    for root, dirs, files in os.walk("checkpoint"):
                        for file in files:
                            myzip.write(os.path.join(root, file))

                # Upload checkpoint to WandB.
                artifact = wandb.Artifact(
                    wandb_run_name,
                    type="algo-checkpoint",
                )
                artifact.add_file("checkpoint.zip")
                wandb.log_artifact(artifact)
                wandb.log(
                    {
                        "videos": [
                            wandb.Video(
                                np.transpose(np.stack(video, axis=0), axes=[0, 3, 1, 2])
                            ) for video in videos
                        ],
                    },
                    step=algo.training_iteration,
                )


if __name__ == "__main__":
    #config = (
    #    DreamerV3Config()
    #        .environment("CartPole-v1")
    #        .training(
    #        model_size="XS",
    #        training_ratio=1024,
    #    )
    #)
    config = SimpleQConfig().environment("CartPole-v1")

    reinforcement_learning_experiment(
        config=config,
        wandb_key="e09303c387ae242600e9d6fd84dbadee35c32101",
        wandb_project="test",
        wandb_run_name="test-run",
        eval_freq=1,
        eval_eps=2,
    )
