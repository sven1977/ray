# This workload tests RLlib's ability to recover from failing workers nodes
import time
import unittest

import ray
from ray._private.test_utils import get_other_nodes
from ray.cluster_utils import Cluster
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.util.state import list_actors


num_redis_shards = 5
redis_max_memory = 10**8
object_store_memory = 10**8
num_nodes = 3


assert (
    num_nodes * object_store_memory + num_redis_shards * redis_max_memory
    < ray._private.utils.get_system_memory() / 2
), (
    "Make sure there is enough memory on this machine to run this "
    "workload. We divide the system memory by 2 to provide a buffer."
)


class NodeFailureTests(unittest.TestCase):
    def start_cluster(self):
        # Simulate a cluster on one machine.
        cluster = Cluster()

        for i in range(num_nodes):
            cluster.add_node(
                redis_port=6379 if i == 0 else None,
                num_redis_shards=num_redis_shards if i == 0 else None,
                num_cpus=2,
                num_gpus=0,
                object_store_memory=object_store_memory,
                redis_max_memory=redis_max_memory,
                dashboard_host="0.0.0.0",
            )
        cluster.wait_for_nodes()
        ray.init(address=cluster.address)
        return cluster

    def stop_cluster(self, cluster):
        ray.shutdown()
        cluster.shutdown()

    def wait_for_all_actors_to_be_back(self):
        # Now, let's wait for Ray to restart all the RolloutWorker actors.
        while True:
            states = [
                a["state"] == "ALIVE"
                for a in list_actors()
                if a["class_name"] == "RolloutWorker"
            ]
            if all(states):
                break
            # Otherwise, wait a bit.
            time.sleep(1)

    def test_node_failures(self):
        """Tests, whether training resumes properly after a worker node failure."""
        for config in [PPOConfig(), APPOConfig()]:
            config = (
                config
                .environment("CartPole-v1")
                .env_runners(
                    num_env_runners=6,
                    validate_env_runners_after_construction=True,
                )
                # Activate EnvRunner fault tolerance.
                .fault_tolerance(recreate_failed_env_runners=True)
                .training(train_batch_size=300)
            )
            print(f"Testing algo={config.algo_class}")
            cluster = self.start_cluster()
            algo = config.build()

            # One step with all nodes up, enough to satisfy resource requirements
            print(algo.train())

            self.assertEqual(algo.workers.num_healthy_remote_workers(), 6)
            self.assertEqual(algo.workers.num_remote_workers(), 6)

            # Remove the first non-head node.
            node_to_kill = get_other_nodes(cluster, exclude_head=True)[0]
            cluster.remove_node(node_to_kill)

            # step() should continue with 4 rollout workers.
            print(algo.train())

            self.assertEqual(algo.workers.num_healthy_remote_workers(), 4)
            self.assertEqual(algo.workers.num_remote_workers(), 6)

            # Node comes back immediately.
            cluster.add_node(
                redis_port=None,
                num_redis_shards=None,
                num_cpus=2,
                num_gpus=0,
                object_store_memory=object_store_memory,
                redis_max_memory=redis_max_memory,
                dashboard_host="0.0.0.0",
            )
            self.wait_for_all_actors_to_be_back()

            # This step should continue with 4 workers, but by the end
            # of weight syncing, the 2 recovered rollout workers should
            # be back.
            print(algo.train())

            # Workers should be back up, everything back to normal.
            self.assertEqual(algo.workers.num_healthy_remote_workers(), 6)
            self.assertEqual(algo.workers.num_remote_workers(), 6)

            algo.stop()
            self.stop_cluster(cluster)


if __name__ == "__main__":
    import pytest
    import sys

    sys.exit(pytest.main(["-v", __file__]))
