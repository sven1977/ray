from ray.rllib.core.models.specs.specs_base import Spec
from ray.rllib.utils.deprecation import Deprecated


@Deprecated(
    help="The Spec checking APIs have been deprecated and cancelled without "
    "replacement.",
    error=True,
)
class SpecDict(dict, Spec):
    pass
