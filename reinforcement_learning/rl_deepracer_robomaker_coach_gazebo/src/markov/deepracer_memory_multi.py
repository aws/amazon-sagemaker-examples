from threading import Thread
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

# Channel used by the training worker to request episodes
WORKER_CHANNEL = 'worker_channel'
# The amount of time to wait before querying the socket
POLL_TIME = 0.001
# Since all the data is handled by the physical memory, there is a limit to the number of steps that can
# be contained in a rollout. This number was determined empirically, as it seems rl_coach is making
# a bunch of hard copies of the transitions
MAX_MEMORY_STEPS = 10000

logger = Logger(__name__, logging.INFO).get_logger()


def log_info(message):
    ''' Helper method that logs the exception
        mesage - Message to send to the log
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

    def __init__(self, params, num_consecutive_playing_steps):
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

    def store(self, obj):
        ''' Stores the data object into the data list along with episode number
            obj - Data object to be stored in the data list
        '''
        self.redis_client.publish(self.params.channel, pickle.dumps((self.current_checkpoint, obj)))

    def set_current_checkpoint(self, checkpoint):
        self.current_checkpoint = checkpoint

    def get_endpoint(self):
        '''Returns a dict with the redis address and port '''
        return get_endpoint_helper(self.params.redis_address, self.params.redis_port)


class PubSubWorkerThread(threading.Thread):
    def __init__(self, sleep_time, memory_backend):
        super(PubSubWorkerThread, self).__init__()
        self.sleep_time = sleep_time
        self._running = threading.Event()
        self.memory_backend = memory_backend

    def run(self):
        if self._running.is_set():
            return
        self._running.set()
        pubsub = self.memory_backend.pubsub
        sleep_time = self.sleep_time
        while self._running.is_set():
            try:
                pubsub.get_message(ignore_subscribe_messages=True,
                                   timeout=sleep_time)
            except Exception as e:
                print(e)
                print("Connection error occured (might be due to output buffer overflow). Reconnecting!")
                self.memory_backend.setup_subscriber()
                pubsub = self.memory_backend.pubsub
        pubsub.close()

    def stop(self):
        # trip the flag so the run loop exits. the run loop will
        # close the pubsub connection, which disconnects the socket
        # and returns the connection to the pool.
        self._running.clear()   



class DeepRacerTrainerBackEnd(MemoryBackend):
    '''Class used by the training worker to retrieve the data from the rollout worker '''

    def __init__(self, params, num_consecutive_playing_steps=20):
        ''' params - Struct containing all the necessary redis parammeters,
                     see RedisPubSubMemoryBackendParame
        '''
        # Redis params
        self.params = params
        self.current_checkpoint = 0
        self.data_queue = queue.Queue(2*num_consecutive_playing_steps)
        # Redis client that will allow us to publish and subscribe to messages
        self.redis_client = redis.Redis(self.params.redis_address, self.params.redis_port)
        self.setup_subscriber()
        self.start_subscriber_thread()
        # Track the total steps taken in the rollout
        self.rollout_steps = 0
        # Episodes in rollout
        self.total_episodes_in_rollout = 0

    def start_subscriber_thread(self):
        self.thread = PubSubWorkerThread(sleep_time=1, memory_backend=self)
        self.thread.start()

    def setup_subscriber(self):
        self.pubsub = self.redis_client.pubsub()
        # Handle data rerurning from the rollout worker via callback
        self.pubsub.subscribe(**{self.params.channel: self.data_handler})
    
    def get_total_episodes_in_rollout(self):
        '''Return the total number of episodes collected in the rollout '''
        return self.total_episodes_in_rollout
    
    def get_rollout_steps(self):
        '''Returns the total number of steps in a rollout '''
        return self.rollout_steps

    def set_current_checkpoint(self, checkpoint):
        self.current_checkpoint = checkpoint

    def data_handler(self, message):
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
                self.data_queue.put_nowait(obj)
        except queue.Full:
            pass
        except Exception as ex:
            log_info("Trainer data handler error: {}".format(ex))

    def fetch(self, num_consecutive_playing_steps=None):
        ''' Retrieves the data from the rollout worker
            num_consecutive_playing_steps - Struct containing the number of episodes to
                                            collect before performing a training iteration
        '''
        episode_counter = 0
        self.request_data = True
        step_counter = 0
        self.rollout_steps = 0
        self.total_episodes_in_rollout = 0
        while episode_counter < num_consecutive_playing_steps.num_steps:
            try:
                obj = self.data_queue.get()
                if obj[0] == self.current_checkpoint and isinstance(obj[1], Episode):
                    episode_counter += 1
                    step_counter += obj[1].length()
                    self.episode_req = episode_counter
                    if step_counter <= MAX_MEMORY_STEPS:
                        self.rollout_steps += obj[1].length()
                        self.total_episodes_in_rollout += 1
                        yield from obj[1]
            except Exception as ex:
                log_info("Trainer fetch error: {}".format(ex))
                continue
        # After the required no. of episodes have been fetched, a new model will be trained. So, increment the checkpoint.
        self.current_checkpoint += 1

    def get_endpoint(self):
        '''Returns a dict with the redis address and port '''
        return get_endpoint_helper(self.params.redis_address, self.params.redis_port)