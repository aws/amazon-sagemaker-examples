import logging
import pickle
import queue
import uuid
from threading import Event, Lock, Thread

import redis
from markov.architecture.constants import NeuralNetwork
from markov.log_handler.logger import Logger
from rl_coach.core_types import Episode
from rl_coach.memories.backend.memory import MemoryBackend
from rl_coach.memories.backend.redis import RedisPubSubMemoryBackendParameters

LOG = Logger(__name__, logging.INFO).get_logger()

# Channel used by the training worker to request episodes
WORKER_CHANNEL = "worker_channel"
# The amount of time to wait before querying the socket
POLL_TIME = 10.0
# Since all the data is handled by the physical memory, there
# is a limit to the number of steps that can
# be contained in a rollout. This number was determined empirically,
# as it seems rl_coach is making
# a bunch of hard copies of the transitions
#
# 3-layer NN (Deep CNN Shallow): 10000 max steps on c4.2xlarge
# 5-layer NN (Deep CNN): 3000 max steps on c4.4xlarge
MAX_MEMORY_STEPS_SHALLOW = 10000
MAX_MEMORY_STEPS = 3000


class DeepRacerRedisPubSubMemoryBackendParameters(RedisPubSubMemoryBackendParameters):
    """
    DeepRacer Redis PubSub Memory Backend Parameters that subclasses RedisPubSubMemoryBackendParameters
    to extend to support num_workers and rollout_idx parameters.
    """

    def __init__(
        self,
        redis_address: str = "",
        redis_port: int = 6379,
        channel: str = "channel-{}".format(uuid.uuid4()),
        orchestrator_params: dict = None,
        run_type="trainer",
        orchestrator_type: str = "kubernetes",
        deployed: str = False,
        num_workers: int = 1,
        rollout_idx: int = 0,
        network_type: str = NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK_SHALLOW.value,
    ):
        super().__init__(
            redis_address=redis_address,
            redis_port=redis_port,
            channel=channel,
            orchestrator_params=orchestrator_params,
            run_type=run_type,
            orchestrator_type=orchestrator_type,
            deployed=deployed,
        )
        self.num_workers = num_workers
        self.rollout_idx = rollout_idx
        self.network_type = network_type


def get_endpoint_helper(redis_address, redis_port):
    """Helper method that returns a dict with the address and port
    redis_address - address to be returned in the dict
    redis_port - Port to be returned in the dict
    """
    return {"redis_address": redis_address, "redis_port": redis_port}


class DeepRacerRolloutBackEnd(MemoryBackend):
    """Class used by the rollout worker to publish data to the training worker"""

    def __init__(self, params, num_consecutive_playing_steps, agent_name):
        """params - Struct containing all the necessary redis parammeters,
                 see RedisPubSubMemoryBackendParameters
        num_consecutive_playing_steps - Struct containing the number of episodes to
                                        collect before performing a training iteration
        """
        self.agent_name = agent_name
        # List of tuples containing the episode number and the episode data
        self.data = list()
        # The last episode number requested by trainer worker
        self.last_request_episode_num = -1
        # The max number of episodes to collect before performing a training iteration
        self.total_episodes = num_consecutive_playing_steps.num_steps
        # Redis params
        self.params = params
        # The episode number of the last episode produced by the rollout worker
        self.last_episode_num = self.params.rollout_idx
        # The total number of rollout worker
        self.num_workers = self.params.num_workers
        # The rollout worker index
        self.rollout_idx = self.params.rollout_idx
        # Redis topic name
        self.topic_name = self.params.channel + "_" + self.agent_name
        # thread lock
        self._lock = Lock()
        # Redis client that will allow us to publish and subscribe to messages
        self.data_client = redis.Redis(self.params.redis_address, self.params.redis_port)
        # Pubsub object that will allow us to subscribe to the data req channels this
        # allow us to get request from the subscriber
        self.data_pubsub = self.data_client.pubsub()
        # Handle request via call back
        self.data_pubsub.subscribe(
            **{WORKER_CHANNEL + "_" + self.agent_name: self.data_req_handler}
        )
        self.data_pubsub.run_in_thread()

    def data_req_handler(self, message):
        """Message handler for training worker request
        message - Request from trainer worker containing the desired episode number
        """
        try:
            episode = pickle.loads(message["data"])

            if episode < 0:
                LOG.info("Negative episode index value")
                return

            with self._lock:
                self.last_request_episode_num = episode
                # Due to the multiple rollout workers, index of self.data array does not reflect actual episode index.
                # The way that episode index (0 based) is distributed throughout rollout workers are as below:
                # - If there are 5 episodes per rollout and 2 rollout workers, then each rollout workers are
                #   responsible for:
                #   - Rollout worker with index 0: 0, 2, 4
                #   - Rollout worker with index 1: 1, 3
                # Thus, (episode < len(self.data) * self.num_workers + self.rollout_idx) condition confirms that
                # this rollout worker already passed the episode index requested, and to confirm that episode
                # is actually executed by this rollout worker, it checks the following condition:
                # - ((episode - self.rollout_idx) % self.num_workers == 0)
                if episode < len(self.data) * self.num_workers + self.rollout_idx:
                    # If episode belongs to current rollout_worker and episode data is ready then,
                    # publish episode data.
                    if (episode - self.rollout_idx) % self.num_workers == 0:
                        # Find actual index in self.data for the requested episode
                        episode_idx_in_data = int((episode - self.rollout_idx) / self.num_workers)
                        self.data_client.publish(
                            self.topic_name, pickle.dumps(self.data[episode_idx_in_data])
                        )

                # If the trainer requests the total episodes we know that the trainer has all the
                # episodes so we will reset the data
                if episode == self.total_episodes:
                    del self.data[:]
                    self.last_episode_num = self.rollout_idx
                    self.last_request_episode_num = -1
                    # rollout worker, simulating last episode, is responsible to send ACK
                    if ((self.total_episodes - 1) - self.rollout_idx) % self.num_workers == 0:
                        # Send an ACK letting the trainer know we have reset the data and it is safe
                        # to train
                        self.data_client.publish(
                            self.topic_name, pickle.dumps((self.total_episodes + 1, ""))
                        )

        except redis.ConnectionError as ex:
            LOG.info("Redis connection error: %s", ex)
        except pickle.PickleError as ex:
            LOG.info("Could not decode/encode trainer request %s", ex)
        except Exception as ex:
            LOG.info("Rollout worker data_req_handler %s", ex)

    def store(self, obj):
        """Stores the data object into the data list along with episode number
        obj - Data object to be stored in the data list
        """
        with self._lock:
            self.data.append((self.last_episode_num, obj))
            # DeepRacerRolloutBackEnd ignores the trainer's request if
            # the data isn't ready at the time. But since we know trainer is waiting
            # send the data as soon as it becomes ready.
            if self.last_episode_num == self.last_request_episode_num:
                episode_idx_in_data = int(
                    (self.last_episode_num - self.rollout_idx) / self.num_workers
                )
                self.data_client.publish(
                    self.topic_name, pickle.dumps(self.data[episode_idx_in_data])
                )
            self.last_episode_num += self.num_workers

    def get_endpoint(self):
        """Returns a dict with the redis address and port"""
        return get_endpoint_helper(self.params.redis_address, self.params.redis_port)


class DeepRacerTrainerBackEnd(MemoryBackend):
    """Class used by the training worker to retrieve the data from the rollout worker"""

    def __init__(self, params, agents_params):
        """params - Struct containing all the necessary redis parammeters,
        see RedisPubSubMemoryBackendParame
        """
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
        self.max_step = (
            MAX_MEMORY_STEPS_SHALLOW
            if self.params.network_type == NeuralNetwork.DEEP_CONVOLUTIONAL_NETWORK_SHALLOW.value
            else MAX_MEMORY_STEPS
        )

        for agent_param in agents_params:
            self.rollout_steps[agent_param.name] = 0
            self.request_events[agent_param.name] = Event()
            self.data_queues[agent_param.name] = queue.Queue(1)
            self.data_pubsubs[agent_param.name] = self.data_client.pubsub()

            # Handle data returning from the rollout worker via callback
            subscriber = (lambda a: lambda m: self.data_handler(m, a))(agent_param.name)
            self.data_pubsubs[agent_param.name].subscribe(
                **{self.params.channel + "_" + agent_param.name: subscriber}
            )
            self.data_pubsubs[agent_param.name].run_in_thread()

            # Use a seperate thread to request data
            publish_worker = (lambda a: lambda: self.publish_worker(a))(agent_param.name)
            Thread(target=publish_worker).start()

    def data_handler(self, message, agent_name):
        """Message handler for data sent from the rollout worker
        message - Tuple sent from the rollout worker containing episode number and data
        """
        try:
            obj = pickle.loads(message["data"])
            if isinstance(obj, tuple):
                self.data_queues[agent_name].put_nowait(obj)
        except queue.Full:
            pass
        except Exception as ex:
            LOG.info("Trainer data handler error: %s", ex)

    def get_rollout_steps(self):
        """Returns the total number of steps in a rollout"""
        return self.rollout_steps

    def get_total_episodes_in_rollout(self):
        """Return the total number of episodes collected in the rollout"""
        return self.total_episodes_in_rollout

    def publish_worker(self, agent_name):
        """Worker responsible for requesting data from the rollout worker"""
        while True:
            try:
                if self.request_data:
                    # Request the desired episode
                    self.data_client.publish(
                        WORKER_CHANNEL + "_" + agent_name, pickle.dumps(self.episode_req)
                    )
                self.request_events[agent_name].wait(POLL_TIME)
                self.request_events[agent_name].clear()
            except redis.ConnectionError as ex:
                LOG.info("Redis connection error: %s : %s", agent_name, ex)
                continue
            except pickle.PickleError as ex:
                LOG.info("Could not decode rollout request %s, %s", agent_name, ex)
                continue
            except Exception as ex:
                LOG.info("Trainer publish worker error: %s, %s", agent_name, ex)
                continue

    def memory_purge(self):
        """Purge Redis Memory"""
        return self.data_client.memory_purge()

    def fetch(self, num_consecutive_playing_steps=None):
        """Retrieves the data from the rollout worker
        num_consecutive_playing_steps - Struct containing the number of episodes to
                                        collect before performing a training iteration
        """
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
                objs = {k: v.get() for k, v in self.data_queues.items()}

                if all(
                    obj[0] == episode_counter and isinstance(obj[1], Episode)
                    for obj in objs.values()
                ):
                    step_counter += sum(obj[1].length() for obj in objs.values())
                    if step_counter <= self.max_step:
                        self.rollout_steps = {
                            k: self.rollout_steps[k] + objs[k][1].length()
                            for k in self.rollout_steps.keys()
                        }
                        self.total_episodes_in_rollout += 1
                        transition_iters = {k: iter(v[1].transitions) for k, v in objs.items()}
                        transition = {k: next(v, None) for k, v in transition_iters.items()}
                        while any(transition.values()):
                            yield transition
                            transition = {k: next(v, None) for k, v in transition_iters.items()}
                    elif episode_counter != num_consecutive_playing_steps.num_steps - 1:
                        # If step_counter goes over self.max_step, then directly request
                        # last episode (index of last episode: num_consecutive_playing_steps.num - 1).
                        # If we just increment the episode one by one till the last one, then it will basically fill up
                        # Redis memory that resides in training worker.
                        # When rollout worker actually returns last episode, then we safely increment episode_counter
                        # to num_consecutive_playing_steps.num, so both rollout worker and training worker can finish
                        # the epoch gracefully.
                        episode_counter = num_consecutive_playing_steps.num_steps - 1
                        self.episode_req = episode_counter
                        continue
                    episode_counter += 1
                    self.episode_req = episode_counter
                # When we request num_consecutive_playing_steps.num we will get back
                # 1 more than the requested index this lets us know the rollout worker
                # has given us all available data
                elif all(
                    obj[0] == num_consecutive_playing_steps.num_steps + 1 for obj in objs.values()
                ):
                    episode_counter = num_consecutive_playing_steps.num_steps + 1
                    self.episode_req = 0
                    self.request_data = False
                    continue
                [event.set() for event in self.request_events.values()]
            except Exception as ex:
                LOG.info("Trainer fetch error: %s", ex)
                continue

    def get_endpoint(self):
        """Returns a dict with the redis address and port"""
        return get_endpoint_helper(self.params.redis_address, self.params.redis_port)
