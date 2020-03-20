import threading
from markov.deepracer_exceptions import GenericRolloutException
from markov.cameras.frustum import Frustum


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
        self.lock = threading.Lock()
        self.camera_namespaces = {}

        # there should be only one camera manager instance
        FrustumManager._instance_ = self

    def add(self, agent_name, observation_list):
        """Add a frustum for given agent with given observation list

        Args:
            agent_name (str): agent_name
            observation_list (list): observation list
        """
        with self.lock:
            self.camera_namespaces[agent_name] = Frustum(agent_name=agent_name,
                                                         observation_list=observation_list)

    def remove(self, agent_name):
        """Remove given agent's frustum from manager.

        Args:
            agent_name (str): agent name
        """
        with self.lock:
            del self.camera_namespaces[agent_name]

    def update(self, agent_name):
        """Update given agent's frustum

        Args:
            agent_name (str): agent name
        """
        with self.lock:
            self.camera_namespaces[agent_name].update()

    def get(self, agent_name):
        """Return given agent's frustum

        Args:
            agent_name (str): agent name
        """
        with self.lock:
            return self.camera_namespaces[agent_name]

