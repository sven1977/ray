from typing import Optional

from ray.rllib.utils.events.observable import Observable
from ray.rllib.utils.typing import EventName


def TriggersEvent(*,
                  name: Optional[EventName] = None,
                  before: bool = True,
                  after: bool = True):

    def _inner(obj):

        def patched(*args, **kwargs):
            # Try to extract self.
            self: Observable = obj.__self__ if hasattr(obj, "__self__") else \
                args[0]
            if not isinstance(self, Observable):
                raise ValueError("`@TriggersEvent` not a valid decorator for "
                                 "non-Observable methods!")

            # Shift out self from args used by triggers as it'll be prepended
            # by `Observable.trigger_event()` anyways.
            trigger_args = args
            if len(args) and self is args[0]:
                trigger_args = args[1:]

            event_base = name or obj.__name__
            # Trigger `before` event passing all args and kwargs as-is to the
            # subscribed event handler(s).
            if before:
                self.trigger_event(f"before_{event_base}", *trigger_args, **kwargs)

            ret = obj(*args, **kwargs)

            # Trigger `after` event passing all args and kwargs as-is plus
            # the return values to the subscribed event handler(s).
            if after:
                try:
                    self.trigger_event(
                        f"after_{event_base}",
                        *trigger_args, **kwargs,
                        return_values=ret)
                # Give user the chance to ignore the auto-appended `return_value`
                # keyword arg.
                except TypeError as e:
                    if "unexpected keyword argument 'return_values'" in e.args[0]:
                        ret = self.trigger_event(f"after_{event_base}", *trigger_args, **kwargs)
                    else:
                        raise e

            # Leave return vlues as-is.
            return ret

        return patched

    return _inner
