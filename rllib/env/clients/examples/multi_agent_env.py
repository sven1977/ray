from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.env.clients.python import EnvClient


if __name__ == "__main__":

    # Create a client object, through which to connect to a server.
    client = EnvClient(
        address="localhost",
        port=8000,
        inference_mode="client",
        # PPO default config object.
        config=PPOConfig(),
    )

    # Use a simple multi-agent env here as an example.
    # However, the env logic here may be anything that runs inside this very
    # python process. It is solely characterized via its interaction with
    # the EnvClient object.
    env_shim = MultiAgentCartPole(config={
        "num_agents": 4,
    })

    # Play through n episodes in sequence.
    num_episodes = 0
    all_done = True
    obs = episode_id = None

    while num_episodes < 3:
        if all_done:
            # Start episode.
            obs = env_shim.reset()
            # Let our client know.
            episode_id = client.start_episode(episode_id=f"episode_{num_episodes}")
            all_done = False

        # Compute actions (this also logs the observation).
        actions = client.compute_action(obs, episode_id=episode_id)

        # Do anything with the actions (here, we need to step our shim env).
        obs, rewards, dones, infos = env_shim.step(actions)

        # Log rewards ..
        client.log_reward(rewards, episode_id=episode_id)
        # .. and infos (if applicable).
        client.log_infos(infos)

        # Check, if any of the agents is done.
        for agent_id, done in dones.items():
            if done:
                client.log_agent_done(agent_id=agent_id, episode_id=episode_id)

        if dones["__all__"]:
            num_episodes += 1
            all_done = True
            # Let client know, episode is done.
            client.end_episode(episode_id=episode_id)
