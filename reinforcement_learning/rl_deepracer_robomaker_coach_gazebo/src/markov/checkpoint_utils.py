import logging
import os
import time
import tensorflow as tf
import glob
import shutil
import re
from markov import utils

from rl_coach.checkpoint import CheckpointStateFile

logger = utils.Logger(__name__, logging.INFO).get_logger()
TEMP_RENAME_FOLDER = "./renamed_checkpoint"

def rename_checkpoints(checkpoint_dir, agent_name):
    ''' Helper method that rename the specific checkpoint in the CheckpointStateFile 
        to be scoped with agent_name
        checkpoint_dir - local checkpoint folder where the checkpoints and .checkpoint file is stored
        agent_name - name of the agent
    '''
    logger.info("Renaming checkpoint from checkpoint_dir: {} for agent: {}".format(checkpoint_dir, agent_name))
    state_file = CheckpointStateFile(os.path.abspath(checkpoint_dir))   
    checkpoint_name = str(state_file.read())
    tf_checkpoint_file = os.path.join(checkpoint_dir, "checkpoint")
    with open(tf_checkpoint_file, "w") as outfile:
        outfile.write("model_checkpoint_path: \"{}\"".format(checkpoint_name))
    
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)
            new_name = var_name
            # Set the new name
            # Replace agent/ or agent_#/ with {agent_name}/  
            new_name = re.sub('agent/|agent_\d+/', '{}/'.format(agent_name), new_name)
            # Rename the variable
            var = tf.Variable(var, name=new_name)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        renamed_checkpoint_path = os.path.join(TEMP_RENAME_FOLDER, checkpoint_name)
        logger.info('Saving updated checkpoint to {}'.format(renamed_checkpoint_path))
        saver.save(sess, renamed_checkpoint_path)
    # Remove the tensorflow 'checkpoint' file
    os.remove(tf_checkpoint_file)
    # Remove the old checkpoint from the checkpoint dir
    for file_name in os.listdir(checkpoint_dir):
        if checkpoint_name in file_name:
            os.remove(os.path.join(checkpoint_dir, file_name))

    # Copy the new checkpoint with renamed variable to the checkpoint dir
    for file_name in os.listdir(TEMP_RENAME_FOLDER):
        full_file_name = os.path.join(os.path.abspath(TEMP_RENAME_FOLDER), file_name)
        if os.path.isfile(full_file_name) and file_name != "checkpoint":
            shutil.copy(full_file_name, checkpoint_dir)

    # Remove files from temp_rename_folder
    shutil.rmtree(TEMP_RENAME_FOLDER)
    
    tf.reset_default_graph()

def modify_checkpoint_variables(checkpoint_dirs, agent_names):
    for checkpoint_dir, agent_name in zip(checkpoint_dirs, agent_names):
        rename_checkpoints(checkpoint_dir, agent_name)

def wait_for_checkpoints(checkpoint_dirs, data_store=None, timeout=10):
    """
    block until there is a checkpoint in all of the checkpoint_dirs
    """
    chkpt_state_files = [CheckpointStateFile(checkpoint_dir) for checkpoint_dir in checkpoint_dirs]
    for i in range(timeout):
        if data_store:
            data_store.load_from_store()
        all_agent_checkpoint_copied = all([chkpt_state_file.read() is not None for chkpt_state_file in chkpt_state_files])
        if all_agent_checkpoint_copied:
            return
        time.sleep(10)

    # one last time
    all_agent_checkpoint_copied = all([chkpt_state_file.read() is not None for chkpt_state_file in chkpt_state_files])
    if all_agent_checkpoint_copied:
        return

    utils.log_and_exit("Checkpoint never found in {} : {}, waited {} seconds." \
                    .format(checkpoint_dirs, all_agent_checkpoint_copied, timeout),
                        utils.SIMAPP_SIMULATION_WORKER_EXCEPTION,
                        utils.SIMAPP_EVENT_ERROR_CODE_500)