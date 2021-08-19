import gym
from gym.spaces import Box, Discrete, MultiDiscrete, Tuple
from typing import Type

from ray.rllib.models.catalog import MODEL_DEFAULTS
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict

tf1, tf, tfv = try_import_tf()


MODELV3_DEFAULTS = dict(MODEL_DEFAULTS, **{
    # Change this to ReLU to stop confusion about our CNN/FC default
    # mismatch.
    "fcnet_activation": "relu",

    # New ModelV3 settings (some of which replace old ones; see del below).
    # Where does this model take its inputs from?
    # - Use None for environment (observations).
    # - Use [str] for another model identifier, whose output this model will use
    #   as input.
    "input_source": None,
    # Do we need to do anything with input (e.g. frame-stacking)?
    # Individual models are allowed to override these `input_requirements`
    # in their c'tors via `self.input_requirements = {...}`.
    # e.g. For frame-stacking the last 4 observation frames, do:
    # input_requirements:
    #     obs:
    #         shift: [-3, -2, -1, 0]
    "input_requirements": None,

    # By default, do NOT add an extra (dense) output layer.
    "output_layer_size": None,
    # But if we do, by default, make it linear.
    "output_layer_activation": None,
})
# Post-fc-net no longer required (keep single models as simple as possible).
# If you need more default model branches, define them in the new "models"
# config key and "snap" them together via the "input_source" key.
del MODELV3_DEFAULTS["post_fcnet_hiddens"]
del MODELV3_DEFAULTS["post_fcnet_activation"]
# Replaced by `output_layer_size`.
del MODELV3_DEFAULTS["no_final_linear"]
# Specify two models, instead (if vf is shared, simply define a common "core").
# No more layer sharing/branching within the same model allowed.
del MODELV3_DEFAULTS["vf_share_layers"]


@ExperimentalAPI("ModelV3")
def build_rllib_default_model(input_space, action_space, name, default_model_config, framework="tf"):
    """
    TODO
    """
    assert framework == "tf", "ERROR: ModelV3 only defined for tf so far!"

    if default_model_config.get("custom_model"):
        raise ValueError(
            "`build_rllib_default_model()` should only be called for creating"
            " default RLlib models! No `custom_model` key allowed within "
            "`default_model_config`!")

        # Validate the given config dict.
        #ModelCatalog._validate_config(config=model_config, framework=framework)

    # Try to get a default v3 model class.
    v3_class = get_v3_model_class(
        input_space, default_model_config, framework=framework)

    if not v3_class:
        raise ValueError(
            "Model class could not be determined! "
            f"`default_model_config`={default_model_config}")

    if default_model_config.get("use_lstm") or \
            default_model_config.get("use_attention"):

        from ray.rllib.models.tf.attention_net import \
            AttentionWrapper, Keras_AttentionWrapper
        from ray.rllib.models.tf.recurrent_net import LSTMWrapper, \
            Keras_LSTMWrapper

        wrapped_cls = v3_class
        if default_model_config.get("use_lstm"):
            v3_class = Keras_LSTMWrapper
            default_model_config["wrapped_cls"] = wrapped_cls
        else:
            v3_class = Keras_AttentionWrapper
            default_model_config["wrapped_cls"] = wrapped_cls

    model = v3_class(
        input_space=input_space,
        action_space=action_space,
        name=name,
        **default_model_config,
    )
    return model


@ExperimentalAPI("ModelV3")
def get_v3_model_class(input_space: gym.Space,
                       model_config: ModelConfigDict,
                       framework: str = "tf") -> Type["tf.keras.Model"]:

    if framework in ["tf2", "tf", "tfe"]:
        from ray.rllib.models.tf.v3.fcnet import FCNet
    else:
        raise ValueError(
            "framework={} not supported in `ModelCatalog._get_v2_model_"
            "class`!".format(framework))

    # Discrete/1D obs-spaces or 2D obs space but traj. view framestacking
    # disabled.
    num_framestacks = model_config.get("num_framestacks", "auto")

    # Tuple space, where at least one sub-space is image.
    # -> Complex input model.
    if isinstance(input_space,
                  Tuple) or (isinstance(input_space, Tuple) and any(
                      isinstance(s, Box) and len(s.shape) >= 2
                      for s in input_space.spaces)):
        raise NotImplementedError("ComplexNet ModelV3!")

    # Single, flattenable/one-hot-able space -> Simple FCNet.
    if isinstance(input_space, (Discrete, MultiDiscrete)) or \
            len(input_space.shape) == 1 or (
            len(input_space.shape) == 2 and (
            num_framestacks == "auto" or num_framestacks <= 1)):
        return FCNet

    raise NotImplementedError("Non-FCNet case!")
