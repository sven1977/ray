.. include:: /_includes/rllib/we_are_hiring.rst

.. include:: /_includes/rllib/new_api_stack.rst


.. _rlmodule-guide:

RL Modules
==========

RLModule is a neural network container that implements three public methods each corresponding to a distinct phase in the reinforcement learning cycle:
:py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.forward_inference` is used to compute actions during an evaluation phase (for example in production),
often requiring greedy or less stochastic action selection.
:py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.forward_exploration` handles the computation of actions during data collection
(if the data is used for a succeeding training step), balancing exploration and exploitation.
Finally, :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.forward_train` manages the training phase, performing calculations required to
compute losses, such as Q-values in a DQN model, value function predictions in a PG-style setup, or world-model predictions in model-based algorithms.


Enabling the RLModule API in the AlgorithmConfig
------------------------------------------------

RL Modules are used exclusively in :ref:`RLlib's new API stack <rllib-new-api-stack-guide>`.

You can activate the new API stack through your :py:class:`~ray.rllib.algorithms.algorithm_config.AlgorithmConfig` instance:

.. literalinclude:: doc_code/rl_module_guide.py
    :language: python
    :start-after: __enabling-rlmodules-in-configs-begin__
    :end-before: __enabling-rlmodules-in-configs-end__


See :ref:`the new API stack migration guide <rllib-new-api-stack-migration-guide>` for more details.


Constructing RL Modules
-----------------------

The :py:class:`~ray.rllib.core.rl_module.rl_module.RLModule` API provides a unified way to define custom reinforcement learning models in RLlib.
This API enables you to design and implement your own neural network models to suit specific needs and supports
highly complex multi-NN setups, often found in multi-agent- or model-based algorithms (or a combination of both).

To maintain consistency and usability, RLlib offers a standardized approach for defining module objects for both single-module
(for example for single-agent) and multi-module use cases (for example for multi-agent learning or other multi-NN setups).

This is achieved through the :py:class:`~ray.rllib.core.rl_module.rl_module.RLModuleSpec` and
:py:class:`~ray.rllib.core.rl_module.multi_rl_module.MultiRLModuleSpec` classes.

.. tab-set::

    .. tab-item:: Single-Module (ex. single-agent)

        .. literalinclude:: doc_code/rl_module_guide.py
            :language: python
            :start-after: __constructing-rlmodules-begin__
            :end-before: __constructing-rlmodules-end__


    .. tab-item:: Multi-Module (ex. multi-agent)

        .. literalinclude:: doc_code/rl_module_guide.py
            :language: python
            :start-after: __constructing-multi-rlmodules-begin__
            :end-before: __constructing-multi-rlmodules-end__


You can pass RL Module specs to the algorithm configuration to be used by the algorithm.

.. tab-set::

    .. tab-item:: Single Agent

        .. literalinclude:: doc_code/rl_module_guide.py
            :language: python
            :start-after: __pass-specs-to-configs-sa-begin__
            :end-before: __pass-specs-to-configs-sa-end__


        .. note::
            For passing RL Module specs, all fields don't have to be filled as they are filled based on the described environment or other algorithm configuration parameters (i.e. ,``observation_space``, ``action_space``, ``model_config_dict`` are not required fields when passing a custom RL Module spec to the algorithm config.)


    .. tab-item:: Multi Agent

        .. literalinclude:: doc_code/rl_module_guide.py
            :language: python
            :start-after: __pass-specs-to-configs-ma-begin__
            :end-before: __pass-specs-to-configs-ma-end__


Writing Custom RL Modules
-------------------------

For single-agent algorithms (e.g., PPO, DQN) or independent multi-agent algorithms (e.g., PPO-MultiAgent), use :py:class:`~ray.rllib.core.rl_module.rl_module.RLModule`. For more advanced multi-agent use cases with a shared communication between agents, extend the :py:class:`~ray.rllib.core.rl_module.multi_rl_module.MultiRLModule` class.

RLlib treats single-agent modules as a special case of :py:class:`~ray.rllib.core.rl_module.multi_rl_module.MultiRLModule` with only one module. Create the multi-agent representation of all RLModules by calling :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.as_multi_rl_module`. For example:

.. literalinclude:: doc_code/rl_module_guide.py
    :language: python
    :start-after: __convert-sa-to-ma-begin__
    :end-before: __convert-sa-to-ma-end__

RLlib implements the following abstract framework specific base classes:

- :class:`TorchRLModule <ray.rllib.core.rl_module.torch_rl_module.TorchRLModule>`: For PyTorch-based RL Modules.
- :class:`TfRLModule <ray.rllib.core.rl_module.tf.tf_rl_module.TfRLModule>`: For TensorFlow-based RL Modules.

The minimum requirement is for sub-classes of :py:class:`~ray.rllib.core.rl_module.rl_module.RLModule` is to implement the following methods:

- :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule._forward_inference`: Forward pass for inference.
- :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule._forward_exploration`: Forward pass for exploration.
- :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule._forward_train`: Forward pass for training.

For your custom :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule._forward_exploration` and :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule._forward_inference`
methods, you must return a dictionary that either contains the key "actions" and/or the key "action_dist_inputs".

If you return the "actions" key:

- RLlib will use the actions provided thereunder as-is.
- If you also returned the "action_dist_inputs" key: RLlib will also create a :py:class:`~ray.rllib.models.distributions.Distribution` object from the distribution parameters under that key and - in the case of :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.forward_exploration` - compute action probs and logp values from the given actions automatically.

If you don't return the "actions" key:

- You must return the "action_dist_inputs" key instead from your :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule._forward_exploration` and :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule._forward_inference` methods.
- RLlib will create a :py:class:`~ray.rllib.models.distributions.Distribution` object from the distribution parameters under that key and sample actions from the thus generated distribution.
- In the case of :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule._forward_exploration`, RLlib will also compute action probs and logp values from the sampled actions automatically.

.. note::

    In the case of :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule._forward_inference`,
    the generated distributions (from returned key "action_dist_inputs") will always be made deterministic first via
    the :py:meth:`~ray.rllib.models.distributions.Distribution.to_deterministic` utility before a possible action sample step.
    For example, sampling from a Categorical distribution will be reduced to selecting the argmax actions from the distribution's logits/probs.

Commonly used distribution implementations can be found under ``ray.rllib.models.torch.torch_distributions`` for torch.
You can choose to compute determinstic actions, by creating the determinstic counterpart of your distribution class through
the :py:meth:`~ray.rllib.models.distributions.Distribution.to_deterministic` method.


.. tab-set::

    .. tab-item:: Returning "actions" key

        .. code-block:: python

            """
            An RLModule whose forward_exploration/inference methods return the
            "actions" key.
            """
            from ray.rllib.core.columns import Columns

            class MyRLModule(TorchRLModule):
                ...

                def _forward_inference(self, batch):
                    ...
                    return {
                        Columns.ACTIONS: ...  # actions will be used as-is
                    }

                def _forward_exploration(self, batch):
                    ...
                    return {
                        Columns.ACTIONS: ...  # actions will be used as-is (no sampling step!)
                        Columns.ACTION_DIST_INPUTS: ...  # optional: If provided, will be used to compute action probs and logp.
                    }

    .. tab-item:: Not returning "actions" key

        .. code-block:: python

            """
            An RLModule whose forward_exploration/inference methods don't return the
            "actions" key.
            """

            class MyRLModule(TorchRLModule):
                ...

                def _forward_inference(self, batch):
                    ...
                    return {
                        # RLlib will:
                        # - Generate distribution from these parameters.
                        # - Convert distribution to a deterministic equivalent.
                        # - "sample" from the deterministic distribution.
                        Columns.ACTION_DIST_INPUTS: ...
                    }

                def _forward_exploration(self, batch):
                    ...
                    return {
                        # RLlib will:
                        # - Generate distribution from these parameters.
                        # - "sample" from the (stochastic) distribution.
                        # - Compute action probs/logs automatically using the sampled
                        #   actions and the generated distribution object.
                        Columns.ACTION_DIST_INPUTS: ...
                    }


Also the :py:class:`~ray.rllib.core.rl_module.rl_module.RLModule` class's constrcutor requires the following arguments:

- :py:attr:`~ray.rllib.core.rl_module.rl_module.RLModule.observation_space`: The observation space of the environment (either processed or raw).
- :py:attr:`~ray.rllib.core.rl_module.rl_module.RLModule.action_space`: The action space of the environment.
- :py:attr:`~ray.rllib.core.rl_module.rl_module.RLModule.inference_only`: Whether the RLModule should be built in inference-only mode leaving out those subcomponents that are only needed for learning.
- :py:attr:`~ray.rllib.core.rl_module.rl_module.RLModule.model_config`: The model config, which is either a custom dictionary (for custom RLModules) or a :py:class:`~ray.rllib.core.rl_module.default_model_config.DefaultModelConfig` dataclass object (only for RLlib's default models). Model hyper-parameters such as number of layers, type of activation, etc. are defined here.
- `catalog_class`: The type of the :py:class:`~ray.rllib.core.models.catalog.Catalog` object to build the RLModule.

When writing RL Modules, you need to use these fields to construct your model.

.. tab-set::

    .. tab-item:: Single Agent (torch)

        .. literalinclude:: doc_code/rl_module_guide.py
            :language: python
            :start-after: __write-custom-sa-rlmodule-torch-begin__
            :end-before: __write-custom-sa-rlmodule-torch-end__


.. In :py:class:`~ray.rllib.core.rl_module.rl_module.RLModule` you can enforce the checking for the existence of certain input or output keys in the data that is communicated into and out of RL Modules. This serves multiple purposes:

.. - For the I/O requirement of each method to be self-documenting.
.. - For failures to happen quickly. If users extend the modules and implement something that does not match the assumptions of the I/O specs, the check reports missing keys and their expected format. For example, RLModule should always have an ``obs`` key in the input batch and an ``action_dist`` key in the output.

.. .. tab-set::

    .. tab-item:: Single Level Keys

        .. literalinclude:: doc_code/rl_module_guide.py
            :language: python
            :start-after: __extend-spec-checking-single-level-begin__
            :end-before: __extend-spec-checking-single-level-end__

    .. tab-item:: Nested Keys

        .. literalinclude:: doc_code/rl_module_guide.py
            :language: python
            :start-after: __extend-spec-checking-nested-begin__
            :end-before: __extend-spec-checking-nested-end__


    .. tab-item:: TensorShape Spec

        .. literalinclude:: doc_code/rl_module_guide.py
            :language: python
            :start-after: __extend-spec-checking-torch-specs-begin__
            :end-before: __extend-spec-checking-torch-specs-end__


    .. tab-item:: Type Spec

        .. literalinclude:: doc_code/rl_module_guide.py
            :language: python
            :start-after: __extend-spec-checking-type-specs-begin__
            :end-before: __extend-spec-checking-type-specs-end__

:py:class:`~ray.rllib.core.rl_module.rl_module.RLModule` has two methods for each forward method, totaling 6 methods that can be override to describe the specs of the input and output of each method:

- :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.input_specs_inference`
- :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.output_specs_inference`
- :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.input_specs_exploration`
- :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.output_specs_exploration`
- :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.input_specs_train`
- :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.output_specs_train`

To learn more, see the `SpecType` documentation.


Writing Custom Multi-Agent RL Modules (Advanced)
------------------------------------------------

For multi-agent modules, RLlib implements :py:class:`~ray.rllib.core.rl_module.multi_rl_module.MultiAgentRLModule`, which is a dictionary of :py:class:`~ray.rllib.core.rl_module.rl_module.RLModule` objects, one for each policy, and possibly some shared modules. The base-class implementation works for most of use cases that need to define independent neural networks for sub-groups of agents. For more complex, multi-agent use cases, where the agents share some part of their neural network, you should inherit from this class and override the default implementation.


The :py:class:`~ray.rllib.core.rl_module.multi_rl_module.MultiRLModule` offers an API for constructing custom models tailored to specific needs. The key method for this customization is :py:meth:`~ray.rllib.core.rl_module.multi_rl_module.MultiRLModule`.build.

The following example creates a custom multi-agent RL module with underlying modules. The modules share an encoder, which gets applied to the global part of the observations space. The local part passes through a separate encoder, specific to each policy.


.. literalinclude:: doc_code/rl_module_guide.py
    :language: python
    :start-after: __write-custom-multirlmodule-shared-enc-begin__
    :end-before: __write-custom-multirlmodule-shared-enc-end__


To construct this custom multi-agent RL module, pass the class to the :py:class:`~ray.rllib.core.rl_module.multi_rl_module.MultiRLModuleSpec` constructor. Also, pass the :py:class:`~ray.rllib.core.rl_module.rl_module.RLModuleSpec` for each agent because RLlib requires the observation, action spaces, and model hyper-parameters for each agent.

.. literalinclude:: doc_code/rl_module_guide.py
    :language: python
    :start-after: __pass-custom-multirlmodule-shared-enc-begin__
    :end-before: __pass-custom-multirlmodule-shared-enc-end__


Extending Existing RLlib RL Modules
-----------------------------------

RLlib provides a number of RL Modules for different frameworks (e.g., PyTorch, TensorFlow, etc.).
To customize existing RLModules you can change the RLModule directly by inheriting the class and changing the
:py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.setup` or other methods.
For example, extend :py:class:`~ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module.PPOTorchRLModule` and augment it with your own customization.
Then pass the new customized class into the appropriate :py:class:`~ray.rllib.algorithms.algorithm_config.AlgorithmConfig`.

There are two possible ways to extend existing RL Modules:

.. tab-set::

    .. tab-item:: Inheriting existing RL Modules

        The default way to extend existing RLModules is to inherit from the framework specific subclasses
        (for example :py:class:`~ray.rllib.core.rl_module.torch.torch_rl_module.TorchRLModule`) and override
        at a minimum the :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.setup` method,
        the
        and any other method you need to customize. Then pass the new customized class into the :py:meth:`~ray.rllib.algorithms.algorithm_config.AlgorithmConfig.rl_module` method
        (`config.rl_module(rl_module_spec=RLModuleSpec(module_class=[your class]))`) to train your custom RLModule.

        .. code-block:: python

            import torch
            nn = torch.nn

            class MyRLModule(TorchRLModule):

                def setup(self):
                    # You have access here to the following already set attributes:
                    self.observation_space
                    self.action_space
                    self.inference_only
                    self.model_config  # <- a dict with custom settings
                    self.catalog
                    ...

                    # Build all the layers and subcomponents here you need for the
                    # RLModule's forward passes.
                    # For example:
                    self._encoder_fcnet = nn.Sequential(...)
                    ...

            # Pass in the custom RL Module class to the spec
            algo_config = algo_config.rl_module(
                rl_module_spec=RLModuleSpec(module_class=MyRLModule)
            )

        A concrete example: If you want to replace the default encoder that RLlib builds for torch, PPO and a given observation space,
        you can override the `__init__` method on the :py:class:`~ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module.PPOTorchRLModule`
        class to create your custom encoder instead of the default one. We do this in the following example.

        .. literalinclude:: ../../../rllib/examples/rl_modules/classes/mobilenet_rlm.py
                :language: python
                :start-after: __sphinx_doc_begin__
                :end-before: __sphinx_doc_end__


    .. tab-item:: Extending RL Module Catalog

        An advanced way to customize your module is by extending its :py:class:`~ray.rllib.core.models.catalog.Catalog`.
        The Catalog is a component that defines the default models and other sub-components for RL Modules based on factors such as ``observation_space``, ``action_space``, etc.
        For more information on the :py:class:`~ray.rllib.core.models.catalog.Catalog` class, refer to the `Catalog user guide <rllib-catalogs.html>`__.
        By modifying the Catalog, you can alter what sub-components are being built for existing RL Modules.
        This approach is useful mostly if you want your custom component to integrate with the decision trees that the Catalogs represent.
        The following use cases are examples of what may require you to extend the Catalogs:

            - Choosing a custom model only for a certain observation space.
            - Using a custom action distribution in multiple distinct Algorithms.
            - Reusing your custom component in many distinct RL Modules.

        For instance, to adapt existing ``PPORLModules`` for a custom graph observation space not supported by RLlib out-of-the-box,
        extend the :py:class:`~ray.rllib.core.models.catalog.Catalog` class used to create the ``PPORLModule``
        and override the method responsible for returning the encoder component to ensure that your custom encoder replaces the default one initially provided by RLlib.

        .. code-block:: python

            class MyAwesomeCatalog(PPOCatalog):

                def build_actor_critic_encoder():
                    # create your awesome graph encoder here and return it
                    pass


            # Pass in the custom catalog class to the spec
            algo_config = algo_config.rl_module(
                rl_module_spec=RLModuleSpec(catalog_class=MyAwesomeCatalog)
            )


Checkpointing RL Modules
------------------------

RL Modules can be checkpointed with their :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.save_to_path` method.
If you have a checkpoint saved and would like to create an RL Module directly from it, use the
:py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.from_checkpoint` method.
If you already have an instantiated RLModule and would like to load a new state (weights) into it from an existing
checkpoint, use the :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule.restore_from_path` method.

The following example shows how these methods can be used outside of, or in conjunction with, an RLlib Algorithm.

.. literalinclude:: doc_code/rl_module_guide.py
        :language: python
        :start-after: __checkpointing-begin__
        :end-before: __checkpointing-end__

Migrating from Custom Policies and Models to RL Modules
-------------------------------------------------------

This document is for those who have implemented custom policies and models in RLlib and want to migrate to the new `~ray.rllib.core.rl_module.rl_module.RLModule` API.
If you have implemented custom policies that extended the `~ray.rllib.policy.eager_tf_policy_v2.EagerTFPolicyV2` or
`~ray.rllib.policy.torch_policy_v2.TorchPolicyV2` classes, you likely did so that you could either modify the behavior of constructing models and distributions
(via overriding `~ray.rllib.policy.torch_policy_v2.TorchPolicyV2.make_model`, `~ray.rllib.policy.torch_policy_v2.TorchPolicyV2.make_model_and_action_dist`), control the action sampling logic (via overriding `~ray.rllib.policy.eager_tf_policy_v2.EagerTFPolicyV2.action_distribution_fn` or `~ray.rllib.policy.eager_tf_policy_v2.EagerTFPolicyV2.action_sampler_fn`), or control the logic for infernce (via overriding `~ray.rllib.policy.policy.Policy.compute_actions_from_input_dict`, `~ray.rllib.policy.policy.Policy.compute_actions`, or `~ray.rllib.policy.policy.Policy.compute_log_likelihoods`). These APIs were built with `ray.rllib.models.modelv2.ModelV2` models in mind to enable you to customize the behavior of those functions. However `~ray.rllib.core.rl_module.rl_module.RLModule` is a more general abstraction that will reduce the amount of functions that you need to override.

In the new `~ray.rllib.core.rl_module.rl_module.RLModule` API the construction of the models and the action distribution class that should be used are best defined in the constructor. That RL Module is constructed automatically if users follow the instructions outlined in the sections `Enabling RL Modules in the Configuration`_ and `Constructing RL Modules`_. `~ray.rllib.policy.policy.Policy.compute_actions` and `~ray.rllib.policy.policy.Policy.compute_actions_from_input_dict` can still be used for sampling actions for inference or exploration by using the ``explore=True|False`` parameter. If called with ``explore=True`` these functions will invoke `~ray.rllib.core.rl_module.rl_module.RLModule.forward_exploration` and if ``explore=False`` then they will call `~ray.rllib.core.rl_module.rl_module.RLModule.forward_inference`.


What your customization could have looked like before:

.. tab-set::

    .. tab-item:: ModelV2

        .. code-block:: python

            from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
            from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2


            class MyCustomModel(TorchModelV2):
                """Code for your previous custom model"""
                ...


            class CustomPolicy(TorchPolicyV2):

                @DeveloperAPI
                @OverrideToImplementCustomLogic
                def make_model(self) -> ModelV2:
                    """Create model.

                    Note: only one of make_model or make_model_and_action_dist
                    can be overridden.

                    Returns:
                    ModelV2 model.
                    """
                    return MyCustomModel(...)


    .. tab-item:: ModelV2 + Distribution


        .. code-block:: python

            from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
            from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2


            class MyCustomModel(TorchModelV2):
                """Code for your previous custom model"""
                ...


            class CustomPolicy(TorchPolicyV2):

                @DeveloperAPI
                @OverrideToImplementCustomLogic
                def make_model_and_action_dist(self):
                    """Create model and action distribution function.

                    Returns:
                        ModelV2 model.
                        ActionDistribution class.
                    """
                    my_model = MyCustomModel(...) # construct some ModelV2 instance here
                    dist_class = ... # Action distribution cls

                    return my_model, dist_class


    .. tab-item:: Sampler functions

        .. code-block:: python

            from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
            from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2

            class CustomPolicy(TorchPolicyV2):

                @DeveloperAPI
                @OverrideToImplementCustomLogic
                def action_sampler_fn(
                    self,
                    model: ModelV2,
                    *,
                    obs_batch: TensorType,
                    state_batches: TensorType,
                    **kwargs,
                ) -> Tuple[TensorType, TensorType, TensorType, List[TensorType]]:
                    """Custom function for sampling new actions given policy.

                    Args:
                        model: Underlying model.
                        obs_batch: Observation tensor batch.
                        state_batches: Action sampling state batch.

                    Returns:
                        Sampled action
                        Log-likelihood
                        Action distribution inputs
                        Updated state
                    """
                    return None, None, None, None


                @DeveloperAPI
                @OverrideToImplementCustomLogic
                def action_distribution_fn(
                    self,
                    model: ModelV2,
                    *,
                    obs_batch: TensorType,
                    state_batches: TensorType,
                    **kwargs,
                ) -> Tuple[TensorType, type, List[TensorType]]:
                    """Action distribution function for this Policy.

                    Args:
                        model: Underlying model.
                        obs_batch: Observation tensor batch.
                        state_batches: Action sampling state batch.

                    Returns:
                        Distribution input.
                        ActionDistribution class.
                        State outs.
                    """
                    return None, None, None


All of the ``Policy.compute_***`` functions expect that
:py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule._forward_exploration` and :py:meth:`~ray.rllib.core.rl_module.rl_module.RLModule._forward_inference`
return a dictionary that either contains the key "actions" and/or the key "action_dist_inputs".

See `Writing Custom Single Agent RL Modules`_ for more details on how to implement your own custom RL Modules.

.. tab-set::

    .. tab-item:: The Equivalent RL Module

        .. code-block:: python

            """
            No need to override any policy functions. Simply instead implement any custom logic in your custom RL Module
            """
            from ray.rllib.models.torch.torch_distributions import YOUR_DIST_CLASS


            class MyRLModule(TorchRLModule):

                def setup(self):
                    # construct any custom networks here using the attributes:
                    # self.observation_space, self.action_space, self.model_config,
                    # self.inference_only, and self.catalog.

                    # Specify an action distribution class here
                    ...

                def _forward_inference(self, batch):
                    ...

                def _forward_exploration(self, batch):
                    ...
