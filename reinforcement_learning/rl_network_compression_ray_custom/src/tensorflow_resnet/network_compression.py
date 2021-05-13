import heapq
import json
import logging
import os
import shutil
import sys

import numpy as np
from tensorflow_resnet.compressor import ModeKeys
from tensorflow_resnet.compressor.core import Fake
from tensorflow_resnet.compressor.layers import LayerState
from tensorflow_resnet.compressor.layers.ops import get_tf_vars_dict
from tensorflow_resnet.compressor.resnet import ResNet18Model
from tensorflow_resnet.compressor.tf_interface import TensorflowInterface as tf
from tensorflow_resnet.dataset import Cifar10

_MASTER_MODEL_NAME = "master"
_MODEL_DIR = "cifar10_model"
_MASTER_DIR = _MODEL_DIR + "/" + _MASTER_MODEL_NAME
_IMPORT_SCOPE = "imported/init"
if os.getenv("SM_HOSTS", None) is None:
    _TEACHER_META_FILE = _MODEL_DIR + "/saved_teacher_weights"
    NETWORK_SAVE_DIR = _MODEL_DIR
    METRICS_SAVE_DIR = _MODEL_DIR
else:
    _TEACHER_META_FILE = "/opt/ml/input/data/training/saved_teacher_weights"
    NETWORK_SAVE_DIR = "/opt/ml/model/"
    METRICS_SAVE_DIR = "/opt/ml/output/data"

_PARAMS = {
    "batch_size": 128,
    "fine_tune": False,
    "remove_layers": None,
    "name": "init",
    "weights": None,
    "dir": _MASTER_DIR,
    "params_scope": _IMPORT_SCOPE,
    "teacher": None,
    # 'learning_rate': 0.01,
}


class NetworkCompressionBase(object):
    """This is a base class that contains the basic methods required as backed for the compression
    environment class. This class contains internal stub methods. The following parameters
    must have been initialized first by the stub class that sub classes it."""

    current_model = None
    master_model = None
    model_class = None
    params = {}

    def _get_observation_of_layer(self, layer, net="master"):
        """This method outputs the observation (description) of the layer requested.
        If needed, it can also return the observation of a pruned network
        Args:
          layer: Return the name of a layer.
        """
        if net == "master":
            desc = self.master_model.net.get_sub_module(str(layer))
        else:
            desc = self.current_model.net.get_sub_module(str(layer))
        if desc is None:
            return [0] * LayerState.LAYER_STATE_LENGTH
        return LayerState.desc2state(desc.get_desc())

    def _get_network_observation(self, net="master"):
        """This method will output the description of the entire network.
        Args:
          net: The network that is used to generate the observation. `master`"""
        out = {}
        for i in range(self.model_class.LAYER_COUNTS[self.model_class.size]):
            out[i] = self._get_observation_of_layer(i, net)
        return out

    def _reset(self):
        """This method essentially resets the network class.  This is needed to clear the old stale
        graphs from memory"""
        inputs = Fake((None, self.dataset.height, self.dataset.width, self.dataset.num_channels))
        self.master_model = self.model_class.builder(inputs, None, ModeKeys.REFERENCE, self.params)
        reference_params = self.params.copy()
        reference_params["name"] = "reference"
        reference_params["teacher"] = self.params["name"]
        self.current_model = self.model_class.builder(
            inputs, None, ModeKeys.REFERENCE, reference_params
        )

    def _perform_actions(self, epochs=5, epochs_between_evals=1):
        """This is a sub-method of the perform actions, that is focussed only on the action itself."""
        logging.info("Performing action: " + self.params["name"])
        self._reset()
        estimator = tf.estimator_builder(
            name=self.params["name"],
            model_params=self.params,
            model=self.model_class,
            remove_layers=self.params["remove_layers"],
            ckpt=None,
        )

        retval = tf.train(
            estimator=estimator,
            dataset=self.dataset,
            epochs=epochs,
            epochs_between_evals=epochs_between_evals,
        )
        return retval

    def _get_observation(self, layer=None, net="master"):
        """Returns the observation of the network
        Args:
          layer: If supplied only outputs that layer, if not, outputs all the network.
          net: If master, it will provide the output for the master network, if not it will
               provide the current network.
        """
        if layer is None:
            return self._get_network_observation(net)
        else:
            if layer > self.get_action_space():
                raise ValueError("Only has " + str(self.get_action_space()) + "layers available.")
            else:
                return self._get_observation_of_layer(layer, net)

    def get_memory_footprint(self, net="current"):
        """returns the memory footprint of all the layers combined for the net requested.

        Args:
         net: Which network to use.
        """
        obs = self._get_observation(None, net)
        m = 0
        for l in obs.keys():
            m += obs[l][-1]
        return m

    def action2remove_layers(self, actions):
        if not actions is None:
            remove_layers = [False] * 40
            nlayers = 0
            for idx, a in enumerate(actions):
                if not isinstance(a, bool):
                    assert a in [0.0, 1.0]
                    if a > 0:
                        remove_layers[idx] = True
                        nlayers += 1
            return remove_layers
        else:
            return actions

    @staticmethod
    def save_params(root, params):
        np.save(root + "/params.pkl", params)

    @staticmethod
    def delete_ckpts_of_action(root):
        try:
            shutil.rmtree(root)
        except Exception as e:
            print(str(e))

    def _update_checkpoint(self):
        if len(self.checkpoints) < self.num_checkpoints:
            heapq.heappush(
                self.checkpoints, (self.reward, self.xfactor, self.accuracy, self.params["dir"])
            )
        elif self.checkpoints[0][0] < self.reward:
            item = heapq.heapreplace(
                self.checkpoints, (self.reward, self.xfactor, self.accuracy, self.params["dir"])
            )
            self.logfile.write("deleting the old checkpoint: " + str(item[-1]) + "\n")
            self.delete_ckpts_of_action(item[-1])
            self.logfile.write("Printing checkpoints ...")
            for item in self.checkpoints:
                self.metricsfile.write(
                    str(item[0])
                    + ","
                    + str(item[1])
                    + ", "
                    + str(item[2])
                    + ", "
                    + str(item[3])
                    + "\n"
                )
            self.metricsfile.flush()
            for item in self.checkpoints:
                self.logfile.write(
                    str(item[0])
                    + ","
                    + str(item[1])
                    + ", "
                    + str(item[2])
                    + ", "
                    + str(item[3])
                    + "\n"
                )
        else:
            return True

        return False


class NetworkCompression(NetworkCompressionBase):
    """This is a stub class containing all the methods that the customer needs to write to
    do channel pruning. For the sake of neatness, other methods have been pushed out into the
    parent class.

    Args:
      network_params: The constructor consumes a params argument that contains all the details that
        are needed to train this network. Below is an example.
        network_params = {
                'resnet_size': _RESNET_SIZE,
                'data_format': 'channels_first',
                'batch_size': _BATCH_SIZE,
                'loss_scale': 1,
                'dtype': 'float32',
                'fine_tune': False,
                'model_class': Net,
                'remove_layers': None,
                'momentum': _MOMENTUM,
                'name': 'teacher',
                'weights': None,
                'params_scope': 'imported/' + _TEACHER_MODEL_NAME
                  }
        model_class: `compressor.resnet.ResNet18Model` like class handlers.
        dataset: `dataset.cifar10.Cifar10` like class handlers.

    """

    ##### Public methods #####
    def __init__(
        self,
        prefix="",
        dataset=Cifar10,
        model_class=ResNet18Model(_MASTER_MODEL_NAME, _PARAMS, scope=_PARAMS["params_scope"]),
        network_params=_PARAMS,
        target_metric=0.81,
    ):
        self.prefix = prefix
        self.dataset = dataset
        self.model_class = model_class
        self.params = model_class.params
        if "remove_layers" in self.params.keys():
            if self.params["remove_layers"] is not None:
                raise ValueError("Intialization should not pass any remove layers argument.")
        self.params["num_classes"] = dataset.num_classes
        self.params["num_images"] = dataset.num_images
        self.params["model_class"] = model_class.definition
        self.params["name"] = self.prefix + "master"
        self.params["distillation_coefficient"] = 1e2

        for key in network_params.keys():
            self.params[key] = network_params[key]
        self._reset()

        self.logfile = open(NETWORK_SAVE_DIR + "/" + self.prefix + "log.log", "a")
        self.metricsfile = open(METRICS_SAVE_DIR + "/" + self.prefix + "_metrics.txt", "a")

        self.master_memory_footprint = self.get_memory_footprint("master")
        self.get_memory_footprint()
        self.target_metric = target_metric
        self.logfile.write("Master accuracy: " + str(self.target_metric) + "\n")
        self.logfile.write("Master memory footprint " + str(self.master_memory_footprint) + "\n")
        self.avg_reward = 0

        num_gpu = int(os.environ.get("SM_NUM_GPUS", 0))
        hosts_info = json.loads(os.environ.get("SM_RESOURCE_CONFIG"))["hosts"]
        num_workers = num_gpu * len(hosts_info)

        # set the number of checkpoints to be stored based on the total number
        # of workers
        self.num_checkpoints = 2
        if num_workers < 3:
            self.num_checkpoints = 5
        elif num_workers < 6:
            self.num_checkpoints = 3
        elif num_workers < 10:
            self.num_checkpoints = 2
        else:
            self.num_checkpoints = 1

        self.checkpoints = []

        self.cache = dict()
        self.cache_str = dict()
        self.actions_taken = 0
        self.reward = 0.5

    def get_action_space(self):
        """Returns the number of layers in the network"""
        return self.model_class.LAYER_COUNTS[self.model_class.size]

    def get_observation_space(self):
        """Returns a tuple of layer number and length of each layer descirptions"""
        return (self.get_action_space(), LayerState.LAYER_STATE_LENGTH)

    def get_current_pos(self):
        """Returns the observation of the master network"""
        obs = self._get_observation(None, "master")
        pos = np.zeros(self.get_observation_space())
        for k in obs.keys():
            pos[k, :] = obs[k]
        return pos

    def perform_actions(self, actions, epochs=5):
        """This will perform the action of removing layers and returns a reward as output. This should
        be called by the step method in the env.

        Args:
          actions: A list if layers to remove in boolean.
          epochs: Number of epochs to train.
        """
        self.params["remove_layers"] = self.action2remove_layers(actions)
        hashcode = hash(str(self.params["remove_layers"]))

        if hashcode in self.cache:
            self.reward = self.cache[hashcode]
            self.actions_taken += 1
            self.logfile.write("cache: " + self.cache_str[hashcode])
            return True

        self.params["name"] = self.prefix + "action_" + str(self.actions_taken)
        self.params["dir"] = NETWORK_SAVE_DIR + "/" + self.params["name"]
        self.params["weights"] = _TEACHER_META_FILE
        self.params["distillation_coefficient"] = 1e2
        self.params["teacher"] = self.prefix + "init"

        b = self.avg_reward / (self.actions_taken + 1)
        C = -1
        try:
            metrics = self._perform_actions(epochs)
            NetworkCompressionBase.save_params(self.params["dir"], self.params)
            C = 1 - float(self.get_memory_footprint()) / self.master_memory_footprint
            self.reward = C * (2 - C)
            self.xfactor = 1.0 / (1 - C)
            self.accuracy = metrics["accuracy"]
            acc_ratio = metrics["accuracy"] / self.target_metric
            self.reward *= acc_ratio
            printstr = "%d: CR: %f, X-factor: %f, Accuracy: %f, Reward: %f" % (
                self.actions_taken,
                C,
                1.0 / (1 - C),
                metrics["accuracy"],
                self.reward,
            )
        except Exception as e:
            self.logfile.write(str(e))
            self.reward = 0
            self.xfactor = 1.0
            self.accuracy = 0.0
            printstr = "%d: CR: %f, %f, Accuracy: %f, Reward: %f" % (
                self.actions_taken,
                C,
                1.0 / (1 - C),
                -1,
                self.reward,
            )

        self.logfile.write(printstr + "\n")
        self.logfile.flush()

        delete_checkpoint = self._update_checkpoint()
        if delete_checkpoint:
            NetworkCompressionBase.delete_ckpts_of_action(self.params["dir"])

        self.actions_taken += 1
        self.cache[hashcode] = self.reward
        self.cache_str[hashcode] = printstr
        self.avg_reward += self.reward

        return True

    def get_reward(self):
        """Returns the value of the current reward."""
        return self.reward

    def reset_params(self, in_params):
        """Accept the params and reset them ."""
        for k in in_params.keys():
            self.params[k] = in_params[k]

    def __del__(self):
        self.logfile.close()
        self.metricsfile.close()
