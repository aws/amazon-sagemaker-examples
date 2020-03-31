from threading import Thread, Event, Lock
import pickle
import time
import queue
import redis
import logging
import threading

from rl_coach.memories.backend.memory import MemoryBackend
from rl_coach.core_types import Episode

from markov.utils import Logger, json_format_logger, build_system_error_dict
from markov.utils import SIMAPP_MEMORY_BACKEND_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500

logger = Logger(__name__, logging.INFO).get_logger()

# Channel used by the training worker to request episodes
WORKER_CHANNEL = 'worker_channel'

# Since all the data is handled by the physical memory, there is a limit to the number of steps that can
# be contained in a rollout. This number was determined empirically, as it seems rl_coach is making
# a bunch of hard copies of the transitions
#
# Cutting down to 5000 from 10000 as the state size is increased:
# - front-facing-camera -> stereo + left_camera + lidar
# - We should be able to handle 6000, but reducing to 5000 to be safe.
# TODO: We need better approach to handle this memory cap.
MAX_MEMORY_STEPS = 5000

def log_info(message):
    ''' Helper method that logs the exception
        message - Message to send to the log
    '''
    json_format_logger(message, **build_system_error_dict(SIMAPP_MEMORY_BACKEND_EXCEPTION,
                                                          SIMAPP_EVENT_ERROR_CODE_500))

def get_endpoint_helper(redis_address, redis_port):
    '''Helper method that returns a dict with the address and port
       redis_address - address to be returned in the dict
       redis_port - Port to be returned in the dict
    '''
    return {'redis_address': redis_address, 'redis_port': redis_port}

class DeepRacerRolloutBackEnd(MemoryBackend):
    ''' Class used by the rollout worker to publish data to the training worker'''

    def __init__(self, params, num_consecutive_playing_steps, agent_name):
        ''' params - Struct containing all the necessary redis parammeters,
                     see RedisPubSubMemoryBackendParameters
            num_consecutive_playing_steps - Struct containing the number of episodes to
                                            collect before performing a training iteration
        '''
        self.current_checkpoint = 0
        # Redis params
        self.params = params
        # Redis client that will allow us to publish and subscribe to messages
        self.redis_client = redis.Redis(
            self.params.redis_address, self.params.redis_port)
        self.agent_name = agent_name

    def store(self, obj):
        ''' Stores the data object into the data list along with episode number
            obj - Data object to be stored in the data list
        '''
        self.redis_client.publish(self.params.channel + '_' + self.agent_name, pickle.dumps((self.current_checkpoint, obj)))

    def set_current_checkpoint(self, checkpoint):
        self.current_checkpoint = checkpoint

    def get_endpoint(self):
        '''Returns a dict with the redis address and port '''
        return get_endpoint_helper(self.params.redis_address, self.params.redis_port)

class PubSubWorkerThread(threading.Thread):
    def __init__(self, sleep_time, memory_backend, agent_name):
        super(PubSubWorkerThread, self).__init__()
        self.sleep_time = sleep_time
        self._running = threading.Event()
        self.memory_backend = memory_backend
        self.agent_name = agent_name

    def run(self):
        if self._running.is_set():
            return
        self._running.set()
        pubsub = self.memory_backend.data_pubsubs[self.agent_name]
        sleep_time = self.sleep_time
        while self._running.is_set():
            try:
                pubsub.get_message(ignore_subscribe_messages=True,
                                   timeout=sleep_time)
            except Exception as e:
                print(e)
                print("Connection error occured (might be due to output buffer overflow). Reconnecting!")
                self.memory_backend.data_pubsubs[self.agent_name].setup_subscriber()
                pubsub = self.memory_backend.data_pubsubs[self.agent_name]
        pubsub.close()

    def stop(self):
        # trip the flag so the run loop exits. the run loop will
        # close the pubsub connection, which disconnects the socket
        # and returns the connection to the pool.
        self._running.clear()

class DeepRacerTrainerBackEnd(MemoryBackend):
    '''Class used by the training worker to retrieve the data from the rollout worker '''

    def __init__(self, params, agents_params):
        ''' params - Struct containing all the necessary redis parammeters,
                     see RedisPubSubMemoryBackendParame
        '''
        # Redis params
        self.params = params
        # Track the total steps taken in the rollout
        self.rollout_steps = dict()
        # Current checkpoint of the training
        self.current_checkpoint = 0
        # Episodes in rollout
        self.total_episodes_in_rollout = 0
        # Queue object to hold data from the rollout worker while waiting to be consumed
        self.data_queues = dict()
        # Redis client that will allow us to publish and subscribe to messages
        self.data_client = redis.Redis(self.params.redis_address, self.params.redis_port)
        # Pubsub object that will allow us to subscribe to the data channel and request data
        self.data_pubsubs = dict()
        
        for agent_param in agents_params:
            self.rollout_steps[agent_param.name] = 0
            num_consecutive_playing_steps = agent_param.algorithm.num_consecutive_playing_steps.num_steps
            self.data_queues[agent_param.name] = queue.Queue(2*num_consecutive_playing_steps)
            self.data_pubsubs[agent_param.name] = self.data_client.pubsub()

            # Handle data returning from the rollout worker via callback in a seperate thread
            subscriber = (lambda a: lambda m: self.data_handler(m, a))(agent_param.name)
            self.data_pubsubs[agent_param.name].subscribe(**{self.params.channel + '_' + agent_param.name: subscriber})
            self.data_pubsubs[agent_param.name].worker_thread = PubSubWorkerThread(sleep_time=1, memory_backend=self, agent_name=agent_param.name)
            self.data_pubsubs[agent_param.name].worker_thread.start()
    
    def set_current_checkpoint(self, checkpoint):
        self.current_checkpoint = checkpoint

    def data_handler(self, message, agent_name):
        ''' Message handler for data sent from the rollout worker
            message - Tuple sent from the rollout worker containing episode number and data
        '''
        try:
            obj = pickle.loads(message['data'])
            if isinstance(obj, tuple) and obj[0] == self.current_checkpoint:
                # Put experiences generated with current model checkpoint on the queue
                # Since this happens in a separate thread, experiences will
                # continuosly be read from the buffer and discarded if not required, possibly avoiding
                # output buffer overflow
                self.data_queues[agent_name].put_nowait(obj)
        except queue.Full:
            pass
        except Exception as ex:
            log_info("Trainer data handler error: {}".format(ex))

    def get_rollout_steps(self):
        '''Returns the total number of steps in a rollout '''
        return self.rollout_steps

    def get_total_episodes_in_rollout(self):
        '''Return the total number of episodes collected in the rollout '''
        return self.total_episodes_in_rollout

    def memory_purge(self):
        '''Purge Redis Memory'''
        return self.data_client.memory_purge()

    def fetch(self, num_consecutive_playing_steps=None):
        ''' Retrieves the data from the rollout worker
            num_consecutive_playing_steps - Struct containing the number of episodes to
                                            collect before performing a training iteration
        '''
        episode_counter = 0
        step_counter = 0
        # Clear any left over Episodes data in queue from previous fetch
        [agent_queue.get() for agent_queue in self.data_queues.values() if not agent_queue.empty()]
        self.rollout_steps = dict.fromkeys(self.rollout_steps, 0)
        self.total_episodes_in_rollout = 0
        while episode_counter < num_consecutive_playing_steps.num_steps:
            try:
                objs = {k: v.get() for k,v in self.data_queues.items()}

                if all(obj[0] == self.current_checkpoint and isinstance(obj[1], Episode) for obj in objs.values()):
                    episode_counter += 1
                    step_counter += sum(obj[1].length() for obj in objs.values())
                    if step_counter <= MAX_MEMORY_STEPS:
                        self.rollout_steps = {k: self.rollout_steps[k] + objs[k][1].length() for k in self.rollout_steps.keys()}
                        self.total_episodes_in_rollout += 1
                        transition_iters = {k: iter(v[1].transitions) for k,v in objs.items()}
                        transition = {k: next(v, None) for k,v in transition_iters.items()}
                        while any(transition.values()):
                            yield transition
                            transition = {k: next(v, None) for k,v in transition_iters.items()}
            except Exception as ex:
                log_info("Trainer fetch error: {}".format(ex))
                continue
        # After the required no. of episodes have been fetched, a new model will be trained. So, increment the checkpoint.
        self.current_checkpoint += 1

    def get_endpoint(self):
        '''Returns a dict with the redis address and port '''
        return get_endpoint_helper(self.params.redis_address, self.params.redis_port)
