from typing import Callable, Optional, Type, Union

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.connectors.learner import (
    AddObservationsFromEpisodesToBatch,
    AddOneTsToEpisodesAndTruncate,
    AddNextObservationsFromEpisodesToTrainBatch,
    GeneralAdvantageEstimation,
)
from ray.rllib.core.learner.learner import Learner
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.execution.rollout_ops import (
    synchronous_parallel_sample,
)
from ray.rllib.execution.train_ops import (
    multi_gpu_train_one_step,
    train_one_step,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import OldAPIStack, override
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.metrics import (
    ALL_MODULES,
    LEARNER_RESULTS,
    LEARNER_UPDATE_TIMER,
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    OFFLINE_SAMPLING_TIMER,
    SAMPLE_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TIMERS,
)
from ray.rllib.utils.typing import (
    EnvType,
    ResultDict,
    RLModuleSpecType,
)
from ray.tune.logger import Logger


class MaxEntIRLConfig(AlgorithmConfig):
    """Defines a configuration class from which a MaxEntIRL Algorithm can be built.

    .. testcode::

        import gymnasium as gym
        import numpy as np

        from pathlib import Path
        from ray.rllib.algorithms.marwil import MARWILConfig

        # Get the base path (to ray/rllib)
        base_path = Path(__file__).parents[2]
        # Get the path to the data in rllib folder.
        data_path = base_path / "tests/data/cartpole/cartpole-v1_large"

        config = MARWILConfig()
        # Enable the new API stack.
        config.api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        # Define the environment for which to learn a policy
        # from offline data.
        config.environment(
            observation_space=gym.spaces.Box(
                np.array([-4.8, -np.inf, -0.41887903, -np.inf]),
                np.array([4.8, np.inf, 0.41887903, np.inf]),
                shape=(4,),
                dtype=np.float32,
            ),
            action_space=gym.spaces.Discrete(2),
        )
        # Set the training parameters.
        config.training(
            beta=1.0,
            lr=1e-5,
            gamma=0.99,
            # We must define a train batch size for each
            # learner (here 1 local learner).
            train_batch_size_per_learner=2000,
        )
        # Define the data source for offline data.
        config.offline_data(
            input_=[data_path.as_posix()],
            # Run exactly one update per training iteration.
            dataset_num_iters_per_learner=1,
        )

        # Build an `Algorithm` object from the config and run 1 training
        # iteration.
        algo = config.build()
        algo.train()

    .. testcode::

        import gymnasium as gym
        import numpy as np

        from pathlib import Path
        from ray.rllib.algorithms.marwil import MARWILConfig
        from ray import train, tune

        # Get the base path (to ray/rllib)
        base_path = Path(__file__).parents[2]
        # Get the path to the data in rllib folder.
        data_path = base_path / "tests/data/cartpole/cartpole-v1_large"

        config = MARWILConfig()
        # Enable the new API stack.
        config.api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        # Print out some default values
        print(f"beta: {config.beta}")
        # Update the config object.
        config.training(
            lr=tune.grid_search([1e-3, 1e-4]),
            beta=0.75,
            # We must define a train batch size for each
            # learner (here 1 local learner).
            train_batch_size_per_learner=2000,
        )
        # Set the config's data path.
        config.offline_data(
            input_=[data_path.as_posix()],
            # Set the number of updates to be run per learner
            # per training step.
            dataset_num_iters_per_learner=1,
        )
        # Set the config's environment for evalaution.
        config.environment(
            observation_space=gym.spaces.Box(
                np.array([-4.8, -np.inf, -0.41887903, -np.inf]),
                np.array([4.8, np.inf, 0.41887903, np.inf]),
                shape=(4,),
                dtype=np.float32,
            ),
            action_space=gym.spaces.Discrete(2),
        )
        # Set up a tuner to run the experiment.
        tuner = tune.Tuner(
            "MARWIL",
            param_space=config,
            run_config=train.RunConfig(
                stop={"training_iteration": 1},
            ),
        )
        # Run the experiment.
        tuner.fit()
    """

    def __init__(self, algo_class=None):
        """Initializes a MaxEntIRLConfig instance."""
        super().__init__(algo_class=algo_class or MaxEntIRL)

        # fmt: off
        # __sphinx_doc_begin__
        # MaxEntIRL specific settings:
        self.train_batch_split_observed_vs_sampled = 0.2

        # Override some of AlgorithmConfig's default values with MARWIL-specific values.

        # You should override input_ to point to an offline dataset
        # (see algorithm.py and algorithm_config.py).
        # The dataset may have an arbitrary number of timesteps
        # (and even episodes) per line.
        # However, each line must only contain consecutive timesteps in
        # order for MARWIL to be able to calculate accumulated
        # discounted returns. It is ok, though, to have multiple episodes in
        # the same line.
        self.input_ = "sampler"
        self.lr = 1e-4
        self.train_batch_size_per_learner = 2000

        # Materialize only the data in raw format, but not the mapped data b/c
        # MARWIL uses a connector to calculate values and therefore the module
        # needs to be updated frequently. This updating would not work if we
        # map the data once at the beginning.
        # TODO (simon, sven): The module is only updated when the OfflinePreLearner
        #   gets reinitiated, i.e. when the iterator gets reinitiated. This happens
        #   frequently enough with a small dataset, but with a big one this does not
        #   update often enough. We might need to put model weigths every couple of
        #   iterations into the object storage (maybe also connector states).
        self.materialize_data = True
        self.materialize_mapped_data = False
        # __sphinx_doc_end__
        # fmt: on

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        train_batch_split_observed_vs_sampled: Optional[float] = NotProvided,
        **kwargs,
    ) -> "MaxEntIRLConfig":
        """Sets the training related configuration.

        Args:
            train_batch_split_observed_vs_sampled: A value in (0.0, 1.0), excluding 0.0
                and 1.0. The share of the entire train batch that's used for computing
                the log likelihood of the trajectories given the rewards from the reward
                model. The share of the train batch used to compute log(Z) is then:
                `1.0 - train_batch_split_observed_vs_sampled`. Chose a smaller value
                here to make sure the log(Z) estimate is based on enough samples.
                The entire train batch always has the size:
                `config.train_batch_size_per_learner`.

        Returns:
            This updated AlgorithmConfig object.
        """
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if train_batch_split_observed_vs_sampled is not NotProvided:
            self.train_batch_split_observed_vs_sampled = (
                train_batch_split_observed_vs_sampled
            )

        return self

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self) -> RLModuleSpecType:
        if self.framework_str == "torch":
            from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule

            return RLModuleSpec(module_class=TorchRLModule)
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Use 'torch' instead."
            )

    @override(AlgorithmConfig)
    def get_default_learner_class(self) -> Union[Type["Learner"], str]:
        if self.framework_str == "torch":
            from ray.rllib.algorithms.maxent_irl.torch.maxent_irl_torch_learner import (
                MaxEntIRLTorchLearner,
            )

            return MaxEntIRLTorchLearner
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Use 'torch' instead."
            )

    @override(AlgorithmConfig)
    def validate(self) -> None:
        # Call super's validation method.
        super().validate()

        # Assert that for a local learner the number of iterations is 1. Note,
        # this is needed because we have no iterators, but instead a single
        # batch returned directly from the `OfflineData.sample` method.
        if (
            self.num_learners == 0
            and not self.dataset_num_iters_per_learner
            and self.enable_rl_module_and_learner
        ):
            self._value_error(
                "When using a local Learner (`config.num_learners=0`), the number of "
                "iterations per learner (`dataset_num_iters_per_learner`) has to be "
                "defined! Set this hyperparameter through `config.offline_data("
                "dataset_num_iters_per_learner=...)`."
            )


class MaxEntIRL(Algorithm):
    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return MaxEntIRLConfig()

    @override(Algorithm)
    def training_step(self) -> None:
        with self.metrics.log_time((TIMERS, OFFLINE_SAMPLING_TIMER)):
            # Sampling from offline data.
            batch_or_iterator = self.offline_data.sample(
                num_samples=self.config.train_batch_size_per_learner,
                num_shards=self.config.num_learners,
                return_iterator=self.config.num_learners > 1,
            )

        with self.metrics.log_time((TIMERS, LEARNER_UPDATE_TIMER)):
            # Updating the reward model.
            # TODO (simon, sven): Check, if we should execute directly s.th. like
            #  `LearnerGroup.update_from_iterator()`.
            learner_results = self.learner_group._update(
                batch=batch_or_iterator,
                minibatch_size=self.config.train_batch_size_per_learner,
                num_iters=self.config.dataset_num_iters_per_learner,
                **self.offline_data.iter_batches_kwargs,
            )

            # Log training results.
            self.metrics.merge_and_log_n_dicts(learner_results, key=LEARNER_RESULTS)
