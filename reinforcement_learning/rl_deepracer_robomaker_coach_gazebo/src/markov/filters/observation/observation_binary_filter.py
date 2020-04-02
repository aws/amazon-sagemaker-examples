from rl_coach.core_types import ObservationType
from rl_coach.filters.observation.observation_filter import ObservationFilter
from rl_coach.spaces import ObservationSpace
from markov.environments.constants import NUMBER_OF_LIDAR_SECTORS, NUMBER_OF_LIDAR_VALUES_IN_EACH_SECTOR, SECTOR_LIDAR_CLIPPING_DIST
import numpy as np
from markov.deepracer_exceptions import GenericRolloutException

class ObservationBinarySectorFilter(ObservationFilter):
    """
    Split the observation space into sectors and categorize the values to binary based on clipping distance
    """
    def __init__(self):
        super().__init__()

    def validate_input_observation_space(self, input_observation_space: ObservationSpace):
        if input_observation_space.shape[0] % NUMBER_OF_LIDAR_VALUES_IN_EACH_SECTOR != 0:
           raise GenericRolloutException("Number of total lidar values is not divisible by number of values in each sector")

    def filter(self, observation: ObservationType, update_internal_state: bool=True) -> ObservationType:
        observation = np.min(observation.reshape(-1, NUMBER_OF_LIDAR_SECTORS), axis=1)
        return (observation < SECTOR_LIDAR_CLIPPING_DIST).astype(float)

    def get_filtered_observation_space(self, input_observation_space: ObservationSpace) -> ObservationSpace:
        input_observation_space.shape[0] = input_observation_space.shape[0] / NUMBER_OF_LIDAR_VALUES_IN_EACH_SECTOR
        return input_observation_space