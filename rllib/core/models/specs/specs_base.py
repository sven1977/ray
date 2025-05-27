import abc
from ray.rllib.utils.deprecation import Deprecated


@Deprecated(
    help="The Spec checking APIs have been deprecated and cancelled without "
    "replacement.",
    error=True,
)
class Spec(abc.ABC):
    pass


@Deprecated(
    help="The Spec checking APIs have been deprecated and cancelled without "
    "replacement.",
    error=True,
)
class TypeSpec(Spec):
    pass


@Deprecated(
    help="The Spec checking APIs have been deprecated and cancelled without "
    "replacement.",
    error=True,
)
class TensorSpec(Spec):
    pass
