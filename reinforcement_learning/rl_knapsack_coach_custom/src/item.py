import numpy as np


class Item:
    def __init__(self, weight=0, volume=0, value=0):
        self.weight = weight
        self.volume = volume
        self.value = value

    @staticmethod
    def get_random_item(max_value, max_weight, max_volume=None):
        weight = np.random.randint(1, max_weight // 10)
        if max_volume:
            volume = np.random.randint(1, max_volume // 10)
        else:
            volume = 0
        value = np.random.randint(1, max_value)
        return Item(weight, volume, value)
