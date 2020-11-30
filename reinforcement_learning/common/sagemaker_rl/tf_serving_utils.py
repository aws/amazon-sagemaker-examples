import ray
import os
import re
import tensorflow

def import_tf1():
    if "2." in tensorflow.__version__[:2]:
        return tensorflow.compat.v1
    return tensorflow


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def change_permissions_recursive(path, mode):
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in [os.path.join(root, d) for d in dirs]:
            os.chmod(dir, mode)
        for file in [os.path.join(root, f) for f in files]:
            os.chmod(file, mode)


def export_tf_serving(agent, output_dir):
    if ray.__version__ >= "0.8.2":
        agent.export_policy_model(os.path.join(output_dir, "1"))
    else:
        tf = import_tf1()
        policy = agent.local_evaluator.policy_map["default"]
        input_signature = {}
        input_signature["observations"] = tf.saved_model.utils.build_tensor_info(policy.observations)

        output_signature = {}
        output_signature["actions"] = tf.saved_model.utils.build_tensor_info(policy.sampler)
        output_signature["logits"] = tf.saved_model.utils.build_tensor_info(policy.logits)

        signature_def = (
            tf.saved_model.signature_def_utils.build_signature_def(
                input_signature, output_signature,
                tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        signature_def_key = (tf.saved_model.signature_constants.
                            DEFAULT_SERVING_SIGNATURE_DEF_KEY)
        signature_def_map = {signature_def_key: signature_def}

        with policy.sess.graph.as_default():
            builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(output_dir, "1"))
            builder.add_meta_graph_and_variables(
                policy.sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map=signature_def_map)
            builder.save()
    print("Saved TensorFlow serving model!")
