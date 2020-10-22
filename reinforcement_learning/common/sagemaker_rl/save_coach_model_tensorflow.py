import shutil
import tensorflow as tf

def save_tf_model():
    ckpt_dir = '/opt/ml/output/data/checkpoint'
    model_dir = '/opt/ml/model'

    # Re-Initialize from the checkpoint so that you will have the latest models up.
    tf.train.init_from_checkpoint(ckpt_dir,
                                    {'main_level/agent/online/network_0/': 'main_level/agent/online/network_0'})
    tf.train.init_from_checkpoint(ckpt_dir,
                                    {'main_level/agent/online/network_1/': 'main_level/agent/online/network_1'})

    # Create a new session with a new tf graph.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())  # initialize the checkpoint.

    # This is the node that will accept the input.
    input_nodes = tf.get_default_graph().get_tensor_by_name('main_level/agent/main/online/' + \
                                                            'network_0/observation/observation:0')
    # This is the node that will produce the output.
    output_nodes = tf.get_default_graph().get_operation_by_name('main_level/agent/main/online/' + \
                                                                'network_1/ppo_head_0/policy')
    # Save the model as a servable model.
    tf.saved_model.simple_save(session=sess,
                                export_dir='model',
                                inputs={"observation": input_nodes},
                                outputs={"policy": output_nodes.outputs[0]})
    # Move to the appropriate folder. Don't mind the directory, this just works.
    # rl-cart-pole is the name of the model. Remember it.
    shutil.move('model/', model_dir + '/model/tf-model/00000001/')
    # EASE will pick it up and upload to the right path.
    print("Success")
