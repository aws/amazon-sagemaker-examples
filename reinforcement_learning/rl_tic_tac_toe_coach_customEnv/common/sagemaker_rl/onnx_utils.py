"""
ONNX Utils to support multiple output heads in agent networks, until future releases of MXNet support this.
"""
import onnx
from onnx import helper, checker, TensorProto


def get_correct_outputs(model):
    """
    Collects the relevent outputs of the model, after identifying the type of RL Agent.
    Currently supports continuous PPO, discrete PPO and DQN agents.
    """
    graph_name = model.graph.output[0].name
    if "_continuousppohead" in graph_name:
        print("ONNX correction applied to continuous PPO agent.")
        return ppo_continuous_outputs(model)
    elif "_discreteppohead" in graph_name:
        print("ONNX correction applied to discrete PPO agent.")
        return ppo_discrete_outputs(model)
    elif "_qhead" in graph_name:
        print("ONNX correction not required for DQN agent.")
        return model.graph.output
    else:
        raise Exception("Can't determine the RL Agent used from the ONNX graph provided.")

        
def make_output(node_name, shape):
    """
    Given a node name and output shape, will construct the correct Protobuf object.
    """
    return helper.make_tensor_value_info(
        name=node_name,
        elem_type=TensorProto.FLOAT,
        shape=shape
    )


def ppo_continuous_outputs(model):
    """
    Collects the output nodes for continuous PPO.
    """
    # determine number of actions 
    log_std_node_name = "generalmodel0_singlemodel1_scaledgradhead0_continuousppohead0_log_std"
    log_std_node = [i for i in model.graph.input if i.name == log_std_node_name][0]
    num_actions = log_std_node.type.tensor_type.shape.dim[0].dim_value
    # identify output nodes
    value_head_name = "generalmodel0_singlemodel0_scaledgradhead0_vhead0_squeeze0"
    value_head = make_output(value_head_name, shape=(1,))
    policy_head_mean_name = "generalmodel0_singlemodel1_scaledgradhead0_continuousppohead0_dense0_fwd"
    policy_head_mean = make_output(policy_head_mean_name, shape=(num_actions,))
    policy_head_std_name = "generalmodel0_singlemodel1_scaledgradhead0_continuousppohead0_broadcast_mul0"
    policy_head_std = make_output(policy_head_std_name, shape=(num_actions,))
    # collect outputs
    output_nodes = [value_head, policy_head_mean, policy_head_std]
    return output_nodes


def ppo_discrete_outputs(model):
    """
    Collects the output nodes for discrete PPO.
    """
    # determine number of actions 
    bias_node_name = "generalmodel0_singlemodel1_scaledgradhead0_discreteppohead0_dense0_bias"
    bias_node = [i for i in model.graph.input if i.name == bias_node_name][0]
    num_actions = bias_node.type.tensor_type.shape.dim[0].dim_value
    # identify output nodes
    value_head_name = "generalmodel0_singlemodel0_scaledgradhead0_vhead0_squeeze0"
    value_head = make_output(value_head_name, shape=(1,))
    policy_head_name = "generalmodel0_singlemodel1_scaledgradhead0_discreteppohead0_softmax0"
    policy_head = make_output(policy_head_name, shape=(num_actions,))
    # collect outputs
    output_nodes = [value_head, policy_head]
    return output_nodes


def save_model(model, output_nodes, filepath):
    """
    Given an in memory model, will save to disk at given filepath.
    """
    new_graph = helper.make_graph(nodes=model.graph.node,
                                  name='new_graph',
                                  inputs=model.graph.input,
                                  outputs=output_nodes,
                                  initializer=model.graph.initializer)
    checker.check_graph(new_graph)
    new_model = helper.make_model(new_graph)
    with open(filepath, "wb") as file_handle:
        serialized = new_model.SerializeToString()
        file_handle.write(serialized)
 

def fix_onnx_model(filepath):
    """
    Applies an inplace fix to ONNX file from Coach. 
    """
    model = onnx.load_model(filepath)
    output_nodes = get_correct_outputs(model)
    save_model(model, output_nodes, filepath)
