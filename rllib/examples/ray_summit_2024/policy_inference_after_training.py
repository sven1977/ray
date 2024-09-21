import gymnasium as gym
import numpy as np

import torch

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.wrappers.atari_wrappers import wrap_atari_for_new_api_stack, wrap_deepmind
from ray.rllib.utils.numpy import convert_to_numpy


checkpoint_path = "/Users/sven/Downloads/checkpoint"
# Create new RLModule and restore its state from the last checkpoint.
rl_module = RLModule.from_checkpoint(checkpoint_path)

# Create the env to do inference in.
env = wrap_atari_for_new_api_stack(
    gym.make(
        "ALE/Pong-v5",
        frameskip=1,
        full_action_space=False,
        repeat_action_probability=0.25,
        render_mode="human",
    ), framestack=4
)
obs, info = env.reset()
env.render()

num_episodes = 0
episode_return = 0.0

while num_episodes < 10:
    # Compute an action using a B=1 observation "batch".
    input_dict = {Columns.OBS: torch.from_numpy(obs).unsqueeze(0)}
    # No exploration.
    rl_module_out = rl_module.forward_inference(input_dict)

    # For discrete action spaces used here, normally, an RLModule "only"
    # produces action logits, from which we then have to sample.
    # However, you can also write custom RLModules that output actions
    # directly, performing the sampling step already inside their
    # `forward_...()` methods.
    logits = convert_to_numpy(rl_module_out[Columns.ACTION_DIST_INPUTS])
    # Act greedily (argmax over logits).
    action = int(np.argmax(logits, axis=-1))
    # Send the computed action to the env.
    obs, reward, terminated, truncated, _ = env.step(action)
    env.render()
    episode_return += reward
    # Is the episode `done`? -> Reset.
    if terminated or truncated:
        print(f"Episode done: Total reward = {episode_return}")
        obs, info = env.reset()
        num_episodes += 1
        episode_return = 0.0
