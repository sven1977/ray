from typing import Any, List, Optional
import uuid

import numpy as np

from ray.util.annotations import PublicAPI
from ray.rllib.policy.sample_batch import SampleBatch


@PublicAPI(stability="alpha")
class SingleAgentEpisode:
    """A class representing a (single-agent) episode from an environment.

    Objects of this class are being built during  of an individual agent and
    store all data that this agent encounters (observations, rewards, infos) as well as
    sends back to the environment (actions, states).

    An Episode can be in one of the following states:
    - Pre-reset: All data lists (e.g. self.observations) are empty and self.t is 0.
    - Post-reset: One observation and one info dict is stored under `self.observations`
      and `self.infos`. self.t is still 0.
    - Ongoing: All data lists contain items, but there is always one more
      observation and info dict than there are actions and rewards. `self.t > 0`.
      `self.is_done is False`.
    - Finished: See "ongoing", but self.is_done returns True, b/c either the environment
      returned a True `truncated` OR `terminated` flag.

    Note that an Epsiode object can also have a `self.t_started` property value of non 0,
    which means the object represents only an Episode chunk. You can concatenate only
    different such Episode chunks that a) have the exact same ID and b) whose `self.t`
    and `self.t_started` match. This makes sure that you won't concatenate two chunks
    that don't immediately succeed each other in the global Episode context.
    An Episode object is considered "complete" if its `self.t_started` is 0 AND its
    `self.is_done` property returns True.
    """
    def __init__(
        self,
        id_: Optional[str] = None,
        *,
        observations: Optional[List[Any]] = None,
        actions: Optional[List[Any]] = None,
        rewards: Optional[List[float]] = None,
        states: Optional[Any] = None,
        t: int = 0,
        is_terminated: bool = False,
        is_truncated: bool = False,
        render_images: Optional[List[np._typing.NDArray]] = None,
    ):
        """Initializes an Episode instance.

        Args:
            id_: The unique ID (str) of this episode. If not provided, will create a
                new unique ID.
            observations: An optional initial list of already seen observations. Note that
                normally, an Episode is created without any data and the very first
                observation created in `self.observations` will be the reset observation
                via `Episode.add_initial_observation()`. Thus,
                the number of observations will always be one larger than the number
                of actions and rewards, unless the episode has not been reset yet (in
                which case all lists are empty).
            actions: An optional initial list of already taken actions.
            TODO: finish docstring
        """
        self.id_ = id_ or uuid.uuid4().hex
        # Observations: t0 (initial obs) to T.
        self.observations = [] if observations is None else observations
        # Actions: t1 to T.
        self.actions = [] if actions is None else actions
        # Rewards: t1 to T.
        self.rewards = [] if rewards is None else rewards
        # h-states: t0 (in case this episode is a continuation chunk, we need to know
        # about the initial h) to T.
        self.states = states
        # The global last timestep of the episode and the timesteps when this chunk
        # started.
        self.t = self.t_started = t
        # obs[-1] is the final observation in the episode.
        self.is_terminated = is_terminated
        # obs[-1] is the last obs in a truncated-by-the-env episode (there will no more
        # observations in following chunks for this episode).
        self.is_truncated = is_truncated
        # RGB uint8 images from rendering the env; the images include the corresponding
        # rewards.
        assert render_images is None or observations is not None
        self.render_images = [] if render_images is None else render_images

    def concat_episode(self, episode_chunk: "SingleAgentEpisode"):
        """Adds the given `episode_chunk` to the right side of self."""
        # Make sure IDs match.
        assert episode_chunk.id_ == self.id_
        # Make sure we are not done yet.
        assert not self.is_done
        # Make sure the timesteps match (our last t should be the same as their first).
        assert self.t == episode_chunk.t_started

        episode_chunk.validate()

        # Make sure, our last observation matches the other episode chunk's first
        # observation.
        assert np.all(episode_chunk.observations[0] == self.observations[-1])
        # Remove our last observation.
        self.observations.pop()

        # Extend ourselves. In case, episode_chunk is already terminated (and numpyfied)
        # we need to convert to lists (as we are ourselves still filling up lists).
        self.observations.extend(list(episode_chunk.observations))
        self.actions.extend(list(episode_chunk.actions))
        self.rewards.extend(list(episode_chunk.rewards))
        self.t = episode_chunk.t
        self.states = episode_chunk.states

        if episode_chunk.is_terminated:
            self.is_terminated = True
        elif episode_chunk.is_truncated:
            self.is_truncated = True
        # Validate.
        self.validate()

    def add_initial_observation(
        self, *, initial_observation, initial_state=None, initial_render_image=None
    ):
        assert not self.is_done
        assert len(self.observations) == 0
        # Assume that this episode is completely empty and has not stepped yet.
        # Leave self.t (and self.t_started) at 0.
        assert self.t == self.t_started == 0

        self.observations.append(initial_observation)
        self.states = initial_state
        if initial_render_image is not None:
            self.render_images.append(initial_render_image)
        self.validate()

    def add_timestep(
        self,
        observation,
        action,
        reward,
        *,
        state=None,
        is_terminated=False,
        is_truncated=False,
        render_image=None,
    ):
        # Cannot add data to an already done episode.
        assert not self.is_done

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.states = state
        self.t += 1
        if render_image is not None:
            self.render_images.append(render_image)
        self.is_terminated = is_terminated
        self.is_truncated = is_truncated
        self.validate()

    def validate(self):
        # Make sure we always have one more obs stored than rewards (and actions)
        # due to the reset and last-obs logic of an MDP.
        assert len(self.observations) == len(self.rewards) + 1 == len(self.actions) + 1
        assert len(self.rewards) == (self.t - self.t_started)

        # Convert all lists to numpy arrays, if we are terminated.
        if self.is_done:
            self.observations = np.array(self.observations)
            self.actions = np.array(self.actions)
            self.rewards = np.array(self.rewards)
            self.render_images = np.array(self.render_images, dtype=np.uint8)

    @property
    def is_done(self):
        """Whether the episode is actually done (terminated or truncated).

        A done episode cannot be continued via `self.add_timestep()` or being
        concatenated on its right-side with another episode chunk or being
        succeeded via `self.create_successor()`.
        """
        return self.is_terminated or self.is_truncated

    def create_successor(self) -> "SingleAgentEpisode":
        """Returns a successor episode chunk (of len=0) continuing with this one.

        The successor will have the same ID and state as self and its only observation
        will be the last observation in self. Its length will therefore be 0 (no
        steps taken yet).

        This method is useful if you would like to discontinue building an episode
        chunk (b/c you have to return it from somewhere), but would like to have a new
        episode (chunk) instance to continue building the actual env episode at a later
        time.

        Returns:
            The successor Episode chunk of this one with the same ID and state and the
            only observation being the last observation in self.
        """
        assert not self.is_done

        return SingleAgentEpisode(
            # Same ID.
            id_=self.id_,
            # First (and only) observation of successor is this episode's last obs.
            observations=[self.observations[-1]],
            # Same state.
            states=self.states,
            # Continue with self's current timestep.
            t=self.t,
        )

    def to_sample_batch(self):
        return SampleBatch(
            {
                SampleBatch.EPS_ID: np.array([self.id_] * len(self)),
                SampleBatch.OBS: self.observations[:-1],
                SampleBatch.NEXT_OBS: self.observations[1:],
                SampleBatch.ACTIONS: self.actions,
                SampleBatch.REWARDS: self.rewards,
                SampleBatch.TERMINATEDS: np.array(
                    [False] * (len(self) - 1) + [self.is_terminated]
                ),
                SampleBatch.TRUNCATEDS: np.array(
                    [False] * (len(self) - 1) + [self.is_truncated]
                ),
            }
        )

    @staticmethod
    def from_sample_batch(batch):
        return SingleAgentEpisode(
            id_=batch[SampleBatch.EPS_ID][0],
            observations=np.concatenate(
                [batch[SampleBatch.OBS], batch[SampleBatch.NEXT_OBS][None, -1]]
            ),
            actions=batch[SampleBatch.ACTIONS],
            rewards=batch[SampleBatch.REWARDS],
            is_terminated=batch[SampleBatch.TERMINATEDS][-1],
            is_truncated=batch[SampleBatch.TRUNCATEDS][-1],
        )

    def get_return(self):
        return sum(self.rewards)

    def get_state(self):
        return list(
            {
                "id_": self.id_,
                "observations": self.observations,
                "actions": self.actions,
                "rewards": self.rewards,
                "states": self.states,
                "t_started": self.t_started,
                "t": self.t,
                "is_terminated": self.is_terminated,
                "is_truncated": self.is_truncated,
            }.items()
        )

    @staticmethod
    def from_state(state):
        eps = SingleAgentEpisode(id_=state[0][1])
        eps.observations = state[1][1]
        eps.actions = state[2][1]
        eps.rewards = state[3][1]
        eps.states = state[4][1]
        eps.t_started = state[5][1]
        eps.t = state[6][1]
        eps.is_terminated = state[7][1]
        eps.is_truncated = state[8][1]
        return eps

    def __len__(self):
        assert len(self.observations) > 0, (
            "ERROR: Cannot determine length of episode that hasn't started yet! "
            "Call `SingleAgentEpisode.add_initial_observation("
            "initial_observation=...)` first "
            "(after which `len(Episode)` will be 0)."
        )
        return len(self.observations) - 1
