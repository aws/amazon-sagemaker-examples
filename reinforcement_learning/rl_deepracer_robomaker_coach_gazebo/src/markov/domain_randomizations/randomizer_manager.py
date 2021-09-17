from markov.log_handler.deepracer_exceptions import GenericRolloutException


class RandomizerManager(object):
    """
    Randomizer Manager class that manages multiple randomizer
    """

    _instance_ = None

    @staticmethod
    def get_instance():
        """Method for getting a reference to the camera manager object"""
        if RandomizerManager._instance_ is None:
            RandomizerManager()
        return RandomizerManager._instance_

    def __init__(self):
        if RandomizerManager._instance_ is not None:
            raise GenericRolloutException("Attempting to construct multiple Randomizer Manager")

        self.randomizers = []

        # there should be only one randomizer manager instance
        RandomizerManager._instance_ = self

    def add(self, randomizer):
        self.randomizers.append(randomizer)

    def randomize(self):
        for randomizer in self.randomizers:
            randomizer.randomize()
