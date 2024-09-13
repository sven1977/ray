from functools import partial

from ray.rllib.connectors.common.add_transformer_input_to_batch import (
    _AddTransformerInputToBatch
)


AddTransformerInputToBatchEnvToModule = partial(
    _AddTransformerInputToBatch, as_learner_connector=False
)
