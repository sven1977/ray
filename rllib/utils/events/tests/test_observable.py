import numpy as np
import sys
import unittest

import ray
from ray.rllib.utils.events.observable import Observable


class Observer:
    def __init__(self):
        self.sum = 0

    def on_add(self, observable, num):
        self.sum += num


class TestObservable(unittest.TestCase):
    """Tests the Observable class and its API."""

    @classmethod
    def setUpClass(cls):
        ray.init(num_cpus=4)

    @classmethod
    def tearDownClass(cls):
        ray.shutdown()

    def test_observable(self):
        observable = Observable()

        # Simple function as subscriber.
        sum_ = 0

        def on_add(observable, num):
            nonlocal sum_
            sum_ += num

        observable.subscribe_to("on_add", on_add)
        observable.trigger_event("on_add", 5)
        self.assertEqual(sum_, 5)
        observable.trigger_event("on_add", 3)
        self.assertEqual(sum_, 8)

        # Lambda as subscriber.
        observable.subscribe_to("on_add", lambda o, n: n)
        observable.trigger_event("on_add", 10)
        self.assertEqual(sum_, 18)

        # Object as subscriber.
        observer = Observer()
        observable.subscribe_to("on_add_2", observer.on_add)
        observable.trigger_event("on_add_2", 5)
        self.assertEqual(observer.sum, 5)
        self.assertEqual(sum_, 18)
        observable.trigger_event("on_add_2", 2)
        self.assertEqual(observer.sum, 7)
        self.assertEqual(sum_, 18)



if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))
