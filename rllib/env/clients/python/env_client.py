import logging
import threading
import time
from typing import Union, Optional
from enum import Enum

import ray.cloudpickle as pickle
from ray.air.checkpoint import Checkpoint
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import (
    AgentID,
    EnvInfoDict,
    EnvObsType,
    EnvActionType,
    EpisodeID,
    MultiAgentDict,
    SampleBatchType,
)

logger = logging.getLogger(__name__)

try:
    import requests  # `requests` is not part of stdlib.
except ImportError:
    requests = None
    logger.warning(
        "Couldn't import `requests` library. Be sure to install it on"
        " the client side."
    )


@PublicAPI
class Commands(Enum):
    # Query the server for a checkpoint or a config.
    # This command is sent after a new connection has been established,
    # in case the client is initialized without config or checkpoint or anything
    # has changed.
    GET_CONFIG_OR_CHECKPOINT = "GET_CONFIG_OR_CHECKPOINT"

    #
    COMPUTE_ACTION = "COMPUTE_ACTION"



@PublicAPI
class EnvClient:
    """REST client to interact with an RLlib server."""

    @PublicAPI
    def __init__(
        self, *,
        address: str,
        port: int,
        config: Optional[AlgorithmConfig] = None,
        checkpoint: Optional[Checkpoint] = None,
        inference_mode: str = "client",
        #NOT NEEDED: update_interval: float = 10.0
    ):
        """Initializes a EnvClient instance.

        Args:
        """
        self.address = address
        self.port = port
        if inference_mode not in ["client", "server"]:
            raise ValueError("`inference_mode` must be either 'client' or 'server'!")
        self.inference_mode = inference_mode

        assert config is not None or checkpoint is not None

        self.config = config
        self.checkpoint = checkpoint

        # Create a policy map from config/checkpoint if provided.
        self.policy_map: PolicyMap = PolicyMap(

        )

    @PublicAPI
    def start_episode(
        self, *, episode_id: Optional[EpisodeID] = None, training_enabled: bool = True
    ) -> str:
        """Record the start of one or more episode(s).

        Args:
            episode_id: Unique string ID for the episode or None for it to
                be auto-assigned.
            training_enabled: Whether to use experiences for this
                episode to improve the policy.

        Returns:
            episode_id: Unique string ID for the episode.
        """

        if self.local:
            self._update_local_policy()
            return self.env.start_episode(episode_id, training_enabled)

        return self._send(
            {
                "episode_id": episode_id,
                "command": Commands.START_EPISODE,
                "training_enabled": training_enabled,
            }
        )["episode_id"]

    @PublicAPI
    def compute_action(
        self,
        *,
        episode_id: Optional[EpisodeID] = None,
        observation: Union[EnvObsType, MultiAgentDict]
    ) -> Union[EnvActionType, MultiAgentDict]:
        """Record an observation and get the (on-policy) action.

        Args:
            episode_id: Episode ID returned from start_episode().
            observation: Current environment observation.

        Returns:
            action: Action from the env action space.
        """

        if self.local:
            self._update_local_policy()
            if isinstance(episode_id, (list, tuple)):
                actions = {
                    eid: self.env.get_action(eid, observation[eid])
                    for eid in episode_id
                }
                return actions
            else:
                return self.env.get_action(episode_id, observation)
        else:
            return self._send(
                {
                    "command": Commands.GET_ACTION,
                    "observation": observation,
                    "episode_id": episode_id,
                }
            )["action"]

    @PublicAPI
    def log_off_policy_action(
        self,
        *,
        input_dict: SampleBatchType,
        action: Union[EnvActionType, MultiAgentDict],
        episode_id: Optional[EpisodeID] = None,
    ) -> None:
        """Record an observation and (off-policy) action taken.

        Args:
            observation: Current environment observation.
            action: Action for the observation.
            episode_id: Episode id returned from start_episode().
        """

        if self.local:
            self._update_local_policy()
            return self.env.log_action(episode_id, observation, action)

        self._send(
            {
                "command": Commands.LOG_ACTION,
                "observation": observation,
                "action": action,
                "episode_id": episode_id,
            }
        )

    @PublicAPI
    def log_reward(
        self,
        *,
        episode_id: Optional[EpisodeID] = None,
        reward: float,
    ) -> None:
        """Record returns from the environment.

        The reward will be attributed to the previous action taken by the
        episode. Rewards accumulate until the next action. If no reward is
        logged before the next action, a reward of 0.0 is assumed.

        Args:
            episode_id: Episode id returned from start_episode().
            reward: Reward from the environment.
        """

        if self.local:
            self._update_local_policy()
            #if multiagent_done_dict is not None:
            #    assert isinstance(reward, dict)
            #    return self.env.log_returns(
            #        episode_id, reward, info, multiagent_done_dict
            #    )
            return self.env.log_returns(episode_id, reward, info)

        self._send(
            {
                "command": Commands.LOG_RETURNS,
                "reward": reward,
                "info": info,
                "episode_id": episode_id,
                "done": multiagent_done_dict,
            }
        )

    @PublicAPI
    def log_agent_done(
        self,
        agent_id: AgentID,
        *,
        episode_id: Optional[EpisodeID] = None,
    ) -> None:
        pass

    @PublicAPI
    def log_infos(self, info: dict, episode_id: Optional[EpisodeID] = None) -> None:
        pass

    @PublicAPI
    def end_episode(
        self, episode_id: Optional[EpisodeID], observation: Union[EnvObsType, MultiAgentDict]
    ) -> None:
        """Record the end of an episode.

        Args:
            episode_id: Episode id returned from start_episode().
            observation: Current environment observation.
        """

        if self.local:
            self._update_local_policy()
            return self.env.end_episode(episode_id, observation)

        self._send(
            {
                "command": Commands.END_EPISODE,
                "observation": observation,
                "episode_id": episode_id,
            }
        )

    def _send(self, data):
        payload = pickle.dumps(data)
        response = requests.post(self.address, data=payload)
        if response.status_code != 200:
            logger.error("Request failed {}: {}".format(response.text, data))
        response.raise_for_status()
        parsed = pickle.loads(response.content)
        return parsed
