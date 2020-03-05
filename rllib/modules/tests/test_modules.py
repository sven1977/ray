from gym.spaces import Dict, Tuple, Box, Discrete
import numpy as np
import unittest

from ray.rllib.experimental.modules import Module
from ray.rllib.utils.numpy import fc
from ray.rllib.utils.framework import try_import_torch, try_import_tf
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.test_utils import check

# Suggested flexible/generic Module (Model) API.
# 1) No difference between tf and torch in constructing the Model.
# 2) Inputs and outputs defined by complex Spaces
#    (which include e.g. hidden states).
# 3) Internal Model `forward` specified by framework-agnostic function.
# 4) Auto-preprocessing available (e.g. flatten of ints and concatenating
#    complex input Spaces)
# 5) Auto-distribution handling (e.g. allow distributions to be added to
#    output or not; distributions can be default ones depending on Space OR
#    custom ones, e.g. mixed-Gaussian instead of Gaussian for a float Box).
# 6) Possibility to stack different Modules together to yield a new Module.
#    (e.g. autoencoders with a distribution in the middle).

tf = try_import_tf()
torch, nn = try_import_torch()


class TestModels(unittest.TestCase):

    def test_model_with_arbitrarily_complex_in_and_out_spaces(self):
        """Tests a policy with complex in- and out-spaces.

        Will auto-create an input concat-layer after one-hotting int components
        and flattening float-components of the input space. After this input
        concat layer comes the "core" network, then the output adapters
        depending on the action space AND the distribution settings
        (output components may or may not have distribution heads).
        """
        # Use an arbitrarily complex input Space.
        input_space = Dict({
            "obs": {
                "img": Box(0, 256, shape=(120, 120, 3), dtype=np.int32),
                "mode": Discrete(4)
            }
        }, main_axes="B")  # <- add batch dimension

        # ... and an arbitrarily complex action Space.
        action_space = Tuple([Discrete(3), Box(-1.0, 1.0, (1,))],
                             main_axes="B")  # <- add batch dimension

        for fw in ["tf", "eager", "torch"]:
            # NOTE: `network` could also be a tf.keras Model OR a
            # torch nn.Module (see test_model_with_custom_network).
            # Here, we use a simple FC net with 2 hiddens (one of which has
            # "tanh" activation).
            pi = from_config(
                Module,
                input_space=input_space, output_space=action_space,
                distributions=True, network=[10, (10, "tanh")],
                framework=fw)
            # Get network parameters.
            params = pi.get_variables()

            # Test action sampling (using the distribution that matches the
            # `action_space`).
            s = input_space.sample(4)  # <- sample a batch of size 4
            a = pi(s)
            self.assertTrue(a.shape[0] == 4)
            self.assertTrue(action_space.contains(a))

            # Test returning the log-likelihood of a given s.
            logp = pi(s, a)
            distribution_params = np.tanh(fc(fc(s, params[], params[]), params[], params[]))
            expected_logp = 0.0 # TODO
            check(logp, expected_logp)

            # Test sampling an action and returning its log-likelihood.
            a, logp = pi(s, log_likelihood=True)
            # TODO

    def test_model_with_custom_network(self):
        input_space = Box((3,), main_axes="B")
        action_space = Discrete(3, main_axes="B")
        for fw in ["tf", "eager", "torch"]:
            # tf.keras model.
            if fw != "torch":
                network = tf.keras.models.Sequential(
                    [tf.keras.layers.Dense(10), tf.keras.layers.Dense(10)])
            # torch nn.Module.
            else:
                network = MyTorchModule(10, 10)

            pi = from_config(
                Module,
                input_space=input_space, output_space=action_space,
                distributions=True, network=network,
                framework=fw)
            # Get network parameters.
            params = pi.get_variables()

            # Test action sampling (using the distribution that matches the
            # `action_space`).
            s = input_space.sample(4)  # <- sample a batch of size 4
            a = pi(s)
            self.assertTrue(a.shape[0] == 4)
            self.assertTrue(action_space.contains(a))

    def test_q_function_on_discrete_actions(self):
        """Tests a Q-function on discrete action space (5 actions)

        Will auto-create an input concat-layer after one-hotting int components
        and flattening float-components of the input space. After this input
        concat layer comes the "core" network, then the automatically generated
        (linear) output layer with 5 nodes (one for each action).
        No distributions.
        """
        num_actions = 5
        input_space = Tuple([Discrete(3), Discrete(5)])

        for fw in ["tf", "eager", "torch"]:
            q_function = from_config(
                Module,
                input_space=input_space, ouptut_space=Discrete(num_actions),
                distributions=False, network=,
                framework=fw)
    
            s = input_space.sample(4)
            # Returns q-values for all `num_actions` actions as a tensor.
            q_vals = q_function(s)
            self.assertTrue(q_vals.shape == (4, num_actions))
            # Returns a single (yet batched) q-value for the given action
            # (given s).
            single_q_value = q_function(s, a)
            self.assertTrue(single_q_value.shape == (4,))

    def test_shared_value_function_policy_net(self):
        """Tests a shared value-function + policy (e.g. see IMPALA)."""
        input_space = Box(shape=(4,), main_axes="B")
        action_space = Discrete(3)  # <- any complex action space possible

        for fw in ["tf", "eager", "torch"]:
            # Create the shared value/policy network.
            pi_and_V = from_config(
                Module,
                input_space=input_space,
                output_space=Dict(a=action_space, V=float),
                distributions=dict(a=True, V=False),
                network=[10, 10],
                framework=fw)
    
            s = input_space.sample(4)
            # Outputs dict with keys "a" and "V", where "a" is an action sample
            # and "V" is the (single - node) value output.
            out = pi_and_V(s)
            # TODO: check `out`.
            # Outputs dict with keys "a" and "V", plus the log-probs for (only!)
            # the given action (a) because V had its distribution set to False.
            a = action_space.sample()
            out, log_probs = pi_and_V(s, dict(a=a))
            # TODO: check `out` and `log_probs`.
            # Outputs dict with keys "a" and "V", where "a" is an action sample and
            # "V" is the (single-node) value output, plus the log-likelihood for
            # `a` (not for `V` b/c V had distribution=False).
            out, log_likelihood = pi_and_V(s, log_likelihood=True)
            # TODO: check `out`.

    def test_dueling_layer_network(self):
        """Tests a dueling layer network arcitecture for dueling DQN.

        Will create two (independent) output-heads after the core network, one
        for Advantages, one for Q-values. The output of the core net is passed
        independently through these two to yield the keys "A" and "V" in the
        output dict.
        """
        input_space = Dict(
            {"a": Box(-1.0, 1.0, shape=(10,)), "b": Discrete(2)},
            main_axes="B")
        action_space = Discrete(5)

        for fw in ["tf", "eager", "torch"]:
            # Define the custom network (including the dueling layer logic).
            # tf.keras
            if tf != "torch":
                network = []
            # torch.nn.Module
            else:
                network = []

            Q = from_config(
                Module,
                input_space=input_space,
                output_space=dict(
                    # Advantages (one per action).
                    A=Box(shape=(action_space.n, ), main_axes="B"),
                    # Q-values (one per action).
                    Q=Box(shape=(action_space.n, ), main_axes="B")
                ),
                distributions=False,
                network=network,
                framework=fw)
                #adapters=dict(A=[hidden keras layer for Advantages], V=[hidden keras layer for Value]))

            s = input_space.sample(3)
            out = Q(s)

            q_vals = out["V"] + out["A"] - 1/|A| * sum(out["A"])
            q_val_for_a = out["V"] + out["A"][a] - 1/|A| * sum(out["A"])
            # NOTE: This dueling formula is only needed for the loss, not for the
            # forward pass. Actions (forward pass) are
            # selected via simply argmax'ing over the advantage values (out["A"]).

    def test_env_dynamics_model_outputting_gaussian_mixture(self):
        """Tests env dynamics model outputting experts of a Gaussian mixture.
        """
        obs_space = Dict(
            {"a": Box(-2.0, 1.0, (5,), dtype=np.int64), "b": Discrete(2)})
        action_space = Discrete(3)

        for fw in ["tf", "eager", "torch"]:
            # Create the Model.
            P = from_config(
                Module,
                input_space=Tuple([obs_space, action_space]),
                output_space=obs_space,
                network=[128],
                # Custom distribution type for output (default would be
                # simple Gaussian).
                distributions={"type": "mixture", "num_experts": 5},
                framework=fw)

            # Returns a next state (s') sample (drawn from the Mixture
            # distribution).
            batch_size = 5
            s_ = P(dict(s=obs_space.sample(batch_size),
                        a=action_space.sample(batch_size)))

            # Return the parameters for the mixture distribution, which can be
            # used in a (supervised) neg-log-likelihood loss function for
            # training P (how likely is the actually observed next state in the
            # predicted next-state distribution?).
            s_parameters = P(dict(s=obs_space.sample(), a=action_space.sample()),
                             parameters_only=True)

    def test_lstm_model(self):
        """An LSTM Model (e.g. IMPALA shared policy/value-network)."""
        action_space = Discrete(4, main_axes="B")
        internal_states_space = Tuple(
            [Box(None, None, (128,)), Box(None, None, (128,))], main_axes="B")
        input_space = Dict(dict(
            prev_a=action_space, prev_r=float,
            s=Dict(dict(
                img=Box(0, 256, shape=(120, 90, 3), dtype=np.int32),
                txt=Discrete(100)
            )),
            # Internal states vector Space.
            h_and_c=internal_states_space
        ), main_axes=["B", "T"])  # <- Setup batch- AND time rank.

        for fw in ["tf", "eager", "torch"]:
            if fw != "torch":
                network = []
            else:
                network = []

            pi_and_V = from_config(
                Module,
                input_space=input_space,
                output_space=dict(
                    a=action_space, h=internal_states_space, V=float),
                network=network,
                # Only `a`-output needs a distribution (here default: Categ.).
                distributions=dict(a=True),
                framework=fw,
            )
            # Resets internal state, but only at batch-position 4.
            pi_and_V.reset_at(4)
            # Resets all internal states (all batch positions).
            pi_and_V.reset()
            # Returns 1) action sample (via distribution), 2) V, and 3) the new
            # internal-state-vector batch.
            a, V, h = pi_and_V(dict(prev_a=, prev_r=, s=[..], h=[internal states vector]))
            # Returns log-prob for given action.
            log_prob = pi_and_V(dict(prev_a=, prev_r=, s=[..], h=[internal states vector]), a)
            # Returns 1) Tuple: action sample, V, final internal state, and
            # 2) log-likelihood of sampled action a (the only one that has a
            # distribution).
            a_V_and_h, log_prob_of_a = pi_and_V(
                dict(prev_a=, prev_r=, s=[..], h = [internalstatesvector]),
                log_likelihood=True)

    def test_autoencoder_model(self):
        """An autoencoder used for creating latent space vectors."""
        encoder = Model(
            input_space=Box(0, 256, shape=(10, 10, 3), dtype=np.int32),
            output_space=Box(shape=(10,), dtype=np.float32),
            distributions=True)
        decoder = Model(
            input_space=encoder.output_space,
            output_space=encoder.input_space,
            distributions=False
        )
        # Unify both to get the complete autoencoder.
        autoencoder = Model.stack([encoder, decoder])

    def test_attention_attention_network_model(self):
        pass
