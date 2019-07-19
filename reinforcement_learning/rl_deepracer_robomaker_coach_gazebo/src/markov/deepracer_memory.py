from threading import Thread
import pickle
import time
import queue
import redis

from rl_coach.memories.backend.memory import MemoryBackend
from rl_coach.core_types import Episode

from markov.utils import Logger, json_format_logger, build_system_error_dict
from markov.utils import SIMAPP_MEMORY_BACKEND_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500

# Channel used by the training worker to request episodes
WORKER_CHANNEL = 'worker_channel'
# The amount of time to wait before querying the socket
POLL_TIME = 0.001

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
        # List of tuples containing the episode number and the episode data
        self.data = list()
        # The episode number of the last episode produced by the rollout worker
        self.last_episode_num = 0
        # The max number of episodes to collect before performing a training iteration
        self.total_episodes = num_consecutive_playing_steps.num_steps
        # Redis params
        self.params = params
        # Redis client that will allow us to publish and subscribe to messages
        self.data_client = redis.Redis(self.params.redis_address, self.params.redis_port)
        # Pubsub object that will allow us to subscribe to the data req channels this
        # allow us to get request from the subscriber
        self.data_pubsub = self.data_client.pubsub()
        # Handle request via call back
        self.data_pubsub.subscribe(**{WORKER_CHANNEL: self.data_req_handler})
        self.data_pubsub.run_in_thread()

    def data_req_handler(self, message):
        ''' Message handler for training worker request
            message - Request from trainer worker containing the desired episode number
        '''
        episode = -1
        try:
            episode = pickle.loads(message['data'])

            if episode < 0:
                log_info("Negative episode index value")
                return

            if episode < len(self.data):
                self.data_client.publish(self.params.channel, pickle.dumps(self.data[episode]))

            # If the trainer requests the total episodes we know that the trainer has all the
            # episodes so we will reset the data
            if episode == self.total_episodes:
                del self.data[:]
                self.last_episode_num = 0
            # Send an ACK letting the trainer know we have reset the data and it is safe
            # to train
                self.data_client.publish(self.params.channel,
                                         pickle.dumps((self.total_episodes + 1, "")))

        except redis.ConnectionError as ex:
            log_info("Redis connection error: {}".format(ex))
        except pickle.PickleError as ex:
            log_info("Could not decode/encode trainer request {}".format(ex))
        except Exception as ex:
            log_info("Rollout worker data_req_handler {}".format(ex))

    def store(self, obj):
        ''' Stores the data object into the data list along with episode number
            obj - Data object to be stored in the data list
        '''
        self.data.append((self.last_episode_num, obj))
        self.last_episode_num += 1

    def get_endpoint(self):
        '''Returns a dict with the redis address and port '''
        return get_endpoint_helper(self.params.redis_address, self.params.redis_port)

class DeepRacerTrainerBackEnd(MemoryBackend):
    '''Class used by the training worker to retrieve the data from the rollout worker '''

    def __init__(self, params):
        ''' params - Struct containing all the necessary redis parammeters,
                     see RedisPubSubMemoryBackendParame
        '''
        # Redis params
        self.params = params
        # Episode number whose data is to be retrieved from the rollout worker
        self.episode_req = 0
        # Queue object to hold data from the rollout worker while waiting to be consumed
        self.data_queue = queue.Queue(1)
        # Flag to notify the publish worker that data should be requested
        self.request_data = False
        # Redis client that will allow us to publish and subscribe to messages
        self.data_client = redis.Redis(self.params.redis_address, self.params.redis_port)
        # Pubsub object that will allow us to subscribe to the data channel and request data
        self.data_pubsub = self.data_client.pubsub()
        # Handle data rerurning from the rollout worker via callback
        self.data_pubsub.subscribe(**{self.params.channel: self.data_handler})
        self.data_pubsub.run_in_thread()
        # Use a seperate thread to request data
        Thread(target=self.publish_worker).start()

    def data_handler(self, message):
        ''' Message handler for data sent from the rollout worker
            message - Tuple sent from the rollout worker containing episode number and data
        '''
        try:
            obj = pickle.loads(message['data'])
            if isinstance(obj, tuple):
                self.data_queue.put_nowait(obj)
        except queue.Full:
            pass
        except Exception as ex:
            log_info("Trainer data handler error: {}".format(ex))

    def publish_worker(self):
        ''' Worker responsible for requesting data from the rollout worker'''
        while True:
            try:
                if self.request_data:
                    # Request the desired episode
                    self.data_client.publish(WORKER_CHANNEL, pickle.dumps(self.episode_req))
                time.sleep(10*POLL_TIME)
            except redis.ConnectionError as ex:
                log_info("Redis connection error: {}".format(ex))
                continue
            except pickle.PickleError as ex:
                log_info("Could not decode rollout request {}".format(ex))
                continue
            except Exception as ex:
                log_info("Trainer publish worker error: {}".format(ex))
                continue

    def fetch(self, num_consecutive_playing_steps=None):
        ''' Retrieves the data from the rollout worker
            num_consecutive_playing_steps - Struct containing the number of episodes to
                                            collect before performing a training iteration
        '''
        episode_counter = 0
        self.request_data = True
        while episode_counter <= num_consecutive_playing_steps.num_steps:
            try:
                obj = self.data_queue.get()
                if obj[0] == episode_counter and isinstance(obj[1], Episode):
                    episode_counter += 1
                    self.episode_req = episode_counter
                    yield from obj[1]
                # When we request num_consecutive_playing_steps.num we will get back
                # 1 more than the requested index this lets us lknow the trollout worker
                # has given us all available data
                elif obj[0] == num_consecutive_playing_steps.num_steps + 1:
                    episode_counter = obj[0]
                    self.episode_req = 0
                    self.request_data = False
            except Exception as ex:
                log_info("Trainer fetch error: {}".format(ex))
                continue

    def get_endpoint(self):
        '''Returns a dict with the redis address and port '''
        return get_endpoint_helper(self.params.redis_address, self.params.redis_port)
