import abc

# Python 2 and 3 compatible Abstract class
ABC = abc.ABCMeta("ABC", (object,), {})


class AbstractRandomizer(ABC):
    """
    Abstract Randomizer class
    """

    def __init__(self):
        pass

    def randomize(self):
        self._randomize()

    @abc.abstractmethod
    def _randomize(self):
        """
        Randomize
        """
        raise NotImplementedError("Randomizer must implement this function")
