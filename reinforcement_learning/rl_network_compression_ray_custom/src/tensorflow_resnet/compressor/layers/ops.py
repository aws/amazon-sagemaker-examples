import pickle
from collections import OrderedDict

import tensorflow as tf


def get_tf_vars_dict(scope=None):
    """Returns all trainable variables in the session in a dictionary form"""
    all_trainable_vars = get_tf_vars_list(scope)
    vars_dict = OrderedDict()
    for var in all_trainable_vars:
        vars_dict[var.name] = var
    return vars_dict


def get_param_from_name(name, scope=None):
    """Returns a particular parameter as a tf element given its name"""
    return get_global_vars_dict(scope)[name]


def load_meta_model_as_np(infile, import_scope="imported"):
    """This will load the meta file into numpy arrays."""
    with tf.Session() as sess:
        restorer = tf.train.import_meta_graph(infile + ".meta", import_scope=import_scope)
        restorer.restore(sess, infile)
        tf_vars = get_global_vars_dict(import_scope)
        np_vars = {}
        for k in tf_vars.keys():
            np_vars[k] = tf_vars[k].eval()

    return np_vars, tf_vars


def load_pkl_obj(name):
    """Loads a pickle model weights for when weights are supplied as initializers to layers"""
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def get_tf_vars_list(scope=None):
    """Returns all the trainable varialbes in the scope as a trainable dictionary."""
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


def get_global_vars_list(scope=None):
    """Returns all the varialbes in the scope as a trainable dictionary."""
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)


def get_global_vars_dict(scope=None):
    """Returns all variables in the session in a dictionary form"""
    all_vars = get_global_vars_list(scope)
    vars_dict = OrderedDict()
    for var in all_vars:
        vars_dict[var.name] = var
    return vars_dict
