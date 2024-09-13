from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core import Columns, DEFAULT_MODULE_ID
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.postprocessing.zero_padding import (
    create_mask_and_seq_lens,
    split_and_zero_pad,
)
from ray.rllib.utils.spaces.space_utils import batch as batch_fn, unbatch
from ray.rllib.utils.typing import EpisodeType
from ray.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class _AddTransformerInputToBatch(ConnectorV2):
    """A connector piece gathering the n most recent observations for an attention net.

    - Should be used only in env-to-module pipelines.
    - Works directly on the incoming episodes list and adds the n most recent
    observations to the batch.
    - This connector does NOT alter the incoming episodes when called.
    #- This connector does NOT work in a `LearnerConnectorPipeline` because it requires
    #the incoming episodes to still be ongoing (in progress) as it only alters the
    #latest observation, not all observations in an episode.

    .. testcode::

        TODO

        import gymnasium as gym
        import numpy as np

        from ray.rllib.connectors.env_to_module import FlattenObservations
        from ray.rllib.env.single_agent_episode import SingleAgentEpisode
        from ray.rllib.utils.test_utils import check

        # Some arbitrarily nested, complex observation space.
        obs_space = gym.spaces.Dict({
            "a": gym.spaces.Box(-10.0, 10.0, (), np.float32),
            "b": gym.spaces.Tuple([
                gym.spaces.Discrete(2),
                gym.spaces.Box(-1.0, 1.0, (2, 1), np.float32),
            ]),
            "c": gym.spaces.MultiDiscrete([2, 3]),
        })
        act_space = gym.spaces.Discrete(2)

        # Two example episodes, both with initial (reset) observations coming from the
        # above defined observation space.
        episode_1 = SingleAgentEpisode(
            observations=[
                {
                    "a": np.array(-10.0, np.float32),
                    "b": (1, np.array([[-1.0], [-1.0]], np.float32)),
                    "c": np.array([0, 2]),
                },
            ],
        )
        episode_2 = SingleAgentEpisode(
            observations=[
                {
                    "a": np.array(10.0, np.float32),
                    "b": (0, np.array([[1.0], [1.0]], np.float32)),
                    "c": np.array([1, 1]),
                },
            ],
        )

        # Construct our connector piece.
        connector = FlattenObservations(obs_space, act_space)

        # Call our connector piece with the example data.
        output_batch = connector(
            rl_module=None,  # This connector works without an RLModule.
            batch={},  # This connector does not alter the input batch.
            episodes=[episode_1, episode_2],
            explore=True,
            shared_data={},
        )

        # The connector does not alter the data and acts as pure pass-through.
        check(output_batch, {})

        # The connector has flattened each item in the episodes to a 1D tensor.
        check(
            episode_1.get_observations(0),
            #         box()  disc(2).  box(2, 1).  multidisc(2, 3)........
            np.array([-10.0, 0.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        )
        check(
            episode_2.get_observations(0),
            #         box()  disc(2).  box(2, 1).  multidisc(2, 3)........
            np.array([10.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        )
    """
    def __init__(
        self,
        input_observation_space: Optional[gym.Space] = None,
        input_action_space: Optional[gym.Space] = None,
        *,
        as_learner_connector: bool = False,
        **kwargs,
    ):
        """Initializes a _AddTransformerInputToBatch instance.

        Args:
            as_learner_connector: Whether this connector is part of a Learner connector
                pipeline, as opposed to an env-to-module pipeline.
        """
        super().__init__(
            input_observation_space=input_observation_space,
            input_action_space=input_action_space,
            **kwargs,
        )
        self._as_learner_connector = as_learner_connector

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        if self._as_learner_connector:
            max_seq_len = rl_module[DEFAULT_MODULE_ID].config.model_config_dict["max_seq_len"]
        else:
            max_seq_len = rl_module.config.model_config_dict["max_seq_len"]

        sa_episodes = list(self.single_agent_episode_iterator(
            episodes,
            # If Learner connector, get all episodes (for train batch).
            # If EnvToModule, get only those ongoing episodes that just had their
            # agent step (b/c those are the ones we need to compute actions for next).
            agents_that_stepped_only=not self._as_learner_connector,
        ))
        T = max(map(len, sa_episodes)) + 1
        if T > max_seq_len:
            T = max_seq_len

        if self._as_learner_connector:
            # OBS are already in batch b/c we hacked the AlgorithmConfig.build_learner_connector() method.
            for column, column_data in batch.copy().items():
                # Do not zero-pad INFOS column.
                if column == Columns.INFOS:
                    continue
                for key, item_list in column_data.items():
                    assert isinstance(key, tuple)
                    if len(key) != 1:
                        raise NotImplementedError
                    column_data[key] = split_and_zero_pad(
                        item_list,
                        max_seq_len=T,
                    )

        for sa_episode in sa_episodes:
            # TODO (sven): We are not supporting multi-agent yet.
            assert sa_episode.module_id is None

            if not self._as_learner_connector:
                # Episode is not finalized yet and thus still operates on lists of items.
                assert not sa_episode.is_finalized

                # If there are enough observations in the episode, use n latest ones.
                if len(sa_episode) + 1 >= T:
                    slice_ = slice(-T, None)
                # If there aren't enough observations, right-zero-pad (get all available
                # ones, from the beginning and let `fill` do its magic on the right side,
                # zero-padding up to n).
                else:
                    slice_ = slice(None, T)
                last_n_obs = sa_episode.get_observations(slice_, fill=0.0)

                # Add the stacked observations (as single batch item) to the batch.
                self.add_batch_item(
                    batch,
                    column=Columns.OBS,
                    item_to_add=batch_fn(last_n_obs),
                    single_agent_episode=sa_episode,
                )

                # Create/update the "loss_mask" (aka. "zero padding mask").
                # Note that we use 0.0 for masked locations (invalid) and 1.0 for valid
                # (non-masked) locations here. This is inverse from the schema used
                # by `torch.nn.TransformerDecoder(memory_key_padding_mask=.)` and thus
                # this mask will have to be inverted in the RLModule's forward pass.
                n_valid = min(T, len(sa_episode) + 1)
                self.add_batch_item(
                    batch,
                    column="transformer_zero_padding",
                    item_to_add=np.array(
                        [1.0] * n_valid + [0.0] * (T - n_valid)
                    ),
                    single_agent_episode=sa_episode,
                )
            # Learner pipeline.
            else:
                # Also, create the loss mask (b/c of our now possibly zero-padded data)
                # as well as the seq_lens array and add these to `data` as well.
                mask, seq_lens = create_mask_and_seq_lens(len(sa_episode), T)
                self.add_n_batch_items(
                    batch=batch,
                    column="transformer_zero_padding",
                    items_to_add=mask,
                    num_items=len(mask),
                    single_agent_episode=sa_episode,
                )
                self.add_n_batch_items(
                    batch=batch,
                    column=Columns.SEQ_LENS,
                    items_to_add=seq_lens,
                    num_items=len(seq_lens),
                    single_agent_episode=sa_episode,
                )

        if "_masks" not in shared_data:
            shared_data["_masks"] = set()
        shared_data["_masks"].add("transformer_zero_padding")

        return batch
