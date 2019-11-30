from threading import Thread, Event, Lock
import pickle
import time
import queue
import redis
import logging

from rl_coach.memories.backend.memory import MemoryBackend
from rl_coach.core_types import Episode

from markov.utils import Logger, json_format_logger, build_system_error_dict
from markov.utils import SIMAPP_MEMORY_BACKEND_EXCEPTION, SIMAPP_EVENT_ERROR_CODE_500

logger = Logger(__name__, logging.INFO).get_logger()

# Channel used by the training worker to request episodes
WORKER_CHANNEL = 'worker_channel'
# The amount of time to wait before querying the socket
POLL_TIME = 10.0
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
        self.agent_name = agent_name
        # List of tuples containing the episode number and the episode data
        self.data = list()
        # The episode number of the last episode produced by the rollout worker
        self.last_episode_num = 0
        # The last episode number requested by trainer worker
        self.last_request_episode_num = -1
        # The max number of episodes to collect before performing a training iteration
        self.total_episodes = num_consecutive_playing_steps.num_steps
        # Redis params
        self.params = params
        # Redis topic name
        self.topic_name = self.params.channel + '_' + self.agent_name
        # thread lock
        self._lock = Lock()
        # Redis client that will allow us to publish and subscribe to messages
        self.data_client = redis.Redis(self.params.redis_address, self.params.redis_port)
        # Pubsub object that will allow us to subscribe to the data req channels this
        # allow us to get request from the subscriber
        self.data_pubsub = self.data_client.pubsub()
        # Handle request via call back
        self.data_pubsub.subscribe(**{WORKER_CHANNEL + '_' + self.agent_name: self.data_req_handler})
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

            with self._lock:
                self.last_request_episode_num = episode
                if episode < len(self.data):
                    self.data_client.publish(self.topic_name,
                                             pickle.dumps(self.data[episode]))

                # If the trainer requests the total episodes we know that the trainer has all the
                # episodes so we will reset the data
                if episode == self.total_episodes:
                    del self.data[:]
                    self.last_episode_num = 0
                    self.last_request_episode_num = -1
                    # Send an ACK letting the trainer know we have reset the data and it is safe
                    # to train
                    self.data_client.publish(self.topic_name,
                                             pickle.dumps((self.total_episodes + 1, "")))

        except redis.ConnectionError as ex:
            logger.info("Redis connection error: {}".format(ex))
        except pickle.PickleError as ex:
            logger.info("Could not decode/encode trainer request {}".format(ex))
        except Exception as ex:
            logger.info("Rollout worker data_req_handler {}".format(ex))

    def store(self, obj):
        ''' Stores the data object into the data list along with episode number
            obj - Data object to be stored in the data list
        '''
        with self._lock:
            self.data.append((self.last_episode_num, obj))
            # DeepRacerRolloutBackEnd ignores the trainer's request if
            # the data isn't ready at the time. But since we know trainer is waiting
            # send the data as soon as it becomes ready.
            if self.last_episode_num <= self.last_request_episode_num:
                self.data_client.publish(self.topic_name,
                                         pickle.dumps(self.data[self.last_episode_num]))
            self.last_episode_num += 1

    def get_endpoint(self):
        '''Returns a dict with the redis address and port '''
        return get_endpoint_helper(self.params.redis_address, self.params.redis_port)


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
        # Episode number whose data is to be retrieved from the rollout worker
        self.episode_req = 0
        # Episodes in rollout
        self.total_episodes_in_rollout = 0
        # Queue object to hold data from the rollout worker while waiting to be consumed
        self.data_queues = dict()
        # Flag to notify the publish worker that data should be requested
        self.request_data = False
        # Redis client that will allow us to publish and subscribe to messages
        self.data_client = redis.Redis(self.params.redis_address, self.params.redis_port)
        # Pubsub object that will allow us to subscribe to the data channel and request data
        self.data_pubsubs = dict()
        self.request_events = dict()

        for agent_param in agents_params:
            self.rollout_steps[agent_param.name] = 0
            self.request_events[agent_param.name] = Event()
            self.data_queues[agent_param.name] = queue.Queue(1)
            self.data_pubsubs[agent_param.name] = self.data_client.pubsub()

            # Handle data returning from the rollout worker via callback
            subscriber = (lambda a: lambda m: self.data_handler(m, a))(agent_param.name)
            self.data_pubsubs[agent_param.name].subscribe(**{self.params.channel + '_' + agent_param.name: subscriber})
            self.data_pubsubs[agent_param.name].run_in_thread()

            # Use a seperate thread to request data
            publish_worker = (lambda a: lambda: self.publish_worker(a))(agent_param.name)
            Thread(target=publish_worker).start()

    def data_handler(self, message, agent_name):
        ''' Message handler for data sent from the rollout worker
            message - Tuple sent from the rollout worker containing episode number and data
        '''
        try:
            obj = pickle.loads(message['data'])
            if isinstance(obj, tuple):
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

    def publish_worker(self, agent_name):
        ''' Worker responsible for requesting data from the rollout worker'''
        while True:
            try:
                if self.request_data:
                    # Request the desired episode
                    self.data_client.publish(WORKER_CHANNEL + '_' + agent_name, pickle.dumps(self.episode_req))
                self.request_events[agent_name].wait(POLL_TIME)
                self.request_events[agent_name].clear()
            except redis.ConnectionError as ex:
                log_info("Redis connection error: {} : {}".format(agent_name, ex))
                continue
            except pickle.PickleError as ex:
                log_info("Could not decode rollout request {}, {}".format(agent_name, ex))
                continue
            except Exception as ex:
                log_info("Trainer publish worker error: {}, {}".format(agent_name, ex))
                continue

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
        self.episode_req = 0
        # Clear any left over Episodes data in queue from previous fetch
        [agent_queue.get() for agent_queue in self.data_queues.values() if not agent_queue.empty()]
        self.request_data = True
        [event.set() for event in self.request_events.values()]
        self.rollout_steps = dict.fromkeys(self.rollout_steps, 0)
        self.total_episodes_in_rollout = 0
        while episode_counter <= num_consecutive_playing_steps.num_steps:
            try:
                objs = {k: v.get() for k,v in self.data_queues.items()}

                if all(obj[0] == episode_counter and isinstance(obj[1], Episode) for obj in objs.values()):
                    episode_counter += 1
                    step_counter += sum(obj[1].length() for obj in objs.values())
                    self.episode_req = episode_counter
                    if step_counter <= MAX_MEMORY_STEPS:
                        self.rollout_steps = {k: self.rollout_steps[k] + objs[k][1].length() for k in self.rollout_steps.keys()}
                        self.total_episodes_in_rollout += 1
                        transition_iters = {k: iter(v[1].transitions) for k,v in objs.items()}
                        transition = {k: next(v, None) for k,v in transition_iters.items()}
                        while any(transition.values()):
                            yield transition
                            transition = {k: next(v, None) for k,v in transition_iters.items()}
                # When we request num_consecutive_playing_steps.num we will get back
                # 1 more than the requested index this lets us know the rollout worker
                # has given us all available data
                elif all(obj[0] == num_consecutive_playing_steps.num_steps + 1 for obj in objs.values()):
                    episode_counter = num_consecutive_playing_steps.num_steps + 1
                    self.episode_req = 0
                    self.request_data = False
                    continue
                [event.set() for event in self.request_events.values()]
            except Exception as ex:
                log_info("Trainer fetch error: {}".format(ex))
                continue

    def get_endpoint(self):
        '''Returns a dict with the redis address and port '''
        return get_endpoint_helper(self.params.redis_address, self.params.redis_port)
