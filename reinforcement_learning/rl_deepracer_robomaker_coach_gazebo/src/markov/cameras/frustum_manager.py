import threading

from markov.cameras.frustum import Frustum
from markov.log_handler.deepracer_exceptions import GenericRolloutException


class FrustumManager(object):
    """
    Frustum Manager class that manages multiple frustum objects
    """

    _instance_ = None

    @staticmethod
    def get_instance():
        """Method for getting a reference to the frustum manager object"""
        if FrustumManager._instance_ is None:
            FrustumManager()
        return FrustumManager._instance_

    def __init__(self):
        if FrustumManager._instance_ is not None:
            raise GenericRolloutException("Attempting to construct multiple frustum manager")
        self.lock = threading.RLock()
        self.frustum_map = {}

        # there should be only one camera manager instance
        FrustumManager._instance_ = self

    def add(self, agent_name, observation_list, version):
        """Add a frustum for given agent with given observation list

        Args:
            agent_name (str): agent_name
            observation_list (list): observation list
            version (float): deepracer physics version
        """
        with self.lock:
            self.frustum_map[agent_name] = Frustum(
                agent_name=agent_name, observation_list=observation_list, version=version
            )

    def remove(self, agent_name):
        """Remove given agent's frustum from manager.

        Args:
            agent_name (str): agent name
        """
        with self.lock:
            del self.frustum_map[agent_name]

    def update(self, agent_name, car_pose):
        """Update given agent's frustum

        Args:
            agent_name (str): agent name
            car_model_state (GetModelState): Gazebo ModelState of the agent
        """
        with self.lock:
            self.frustum_map[agent_name].update(car_pose)

    def get(self, agent_name):
        """Return given agent's frustum

        Args:
            agent_name (str): agent name
        """
        with self.lock:
            return self.frustum_map[agent_name]
