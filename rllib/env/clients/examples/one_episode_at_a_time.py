import gym

from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.clients.python import EnvClient


if __name__ == "__main__":

    # Create a client object, through which to connect to a server.
    client = EnvClient(
        address="localhost",
        port=8000,
        inference_mode="client",
        # DQN default config object.
        config=DQNConfig(),
    )

    # Use a simple gym env here as an example.
    # However, the env logic here may be anything that runs inside this very
    # python process. It is solely characterized via its interaction with
    # the EnvClient object.
    env_shim = gym.make("FrozenLake-v1")

    # Play through n episodes in sequence, not using episode IDs (automatically
    # created/inferred).
    num_episodes = 0
    done = True
    obs = None

    while num_episodes < 3:
        if done:
            # Start episode.
            obs = env_shim.reset()
            # Let our client know.
            client.start_episode()

        # Compute actions (this also logs the observation).
        action = client.compute_action(obs)

        # Do anything with the actions (here, we need to step our shim env).
        obs, reward, done, infos = env_shim.step(action)

        # Log rewards ..
        client.log_reward(reward)
        # .. and infos (if applicable).
        client.log_infos(infos)

        # Check, if the episode is done.
        if done:
            num_episodes += 1
            # Let client know, episode is done.
            client.end_episode()
