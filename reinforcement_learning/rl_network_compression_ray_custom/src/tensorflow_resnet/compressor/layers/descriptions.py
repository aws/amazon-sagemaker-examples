class LayerState:
    """
    This class has a lot of static methods that basically convert the layer descriptions into one
    that is readable by the NetworkCompression system. These convert everything to integers. The last element of
    this description is always the memory footprint
    """

    LAYER_STATE_LENGTH = 8  # length of the layer description.
    LAYER_IDS = {  # each layer gets its own integer in description.
        "Conv": 1,
        "Dense": 2,
        "ReLU": 3,
        "BatchNorm": 4,
        "Pool": 5,
    }

    @staticmethod
    def desc2state(desc):  # Global method for converting description to state.
        if desc[2] == "Conv":
            return LayerState.conv_state(desc)
        elif desc[2] == "Dense":
            return LayerState.dense_state(desc)
        elif desc[2] == "ReLU":
            return LayerState.relu_state(desc)
        elif desc[2] == "BatchNorm":
            return LayerState.bn_state(desc)
        elif desc[2] == "Pool":
            return LayerState.pool_state(desc)

    @staticmethod
    def conv_state(desc):
        state = [0] * LayerState.LAYER_STATE_LENGTH
        state[0] = desc[0]  # start
        state[1] = desc[1]  # end
        state[2] = LayerState.LAYER_IDS[desc[2]]  # Layer id.
        state[3] = desc[3]  # filters
        state[4] = desc[4]  # kernel Size
        state[5] = desc[5]  # strides
        if desc[6] == "SAME":
            state[6] = 1
        else:
            state[6] = 0
        state[7] = desc[7]  # memory footprint
        return state

    @staticmethod
    def dense_state(desc):
        state = [0] * LayerState.LAYER_STATE_LENGTH
        state[0] = desc[0]  # start
        state[1] = desc[1]  # end
        state[2] = LayerState.LAYER_IDS[desc[2]]  # Layer id.
        state[3] = desc[3]  # num_units
        state[7] = desc[4]  # memory footprint
        return state

    @staticmethod
    def bn_state(desc):
        state = [0] * LayerState.LAYER_STATE_LENGTH
        state[0] = desc[0]  # start
        state[1] = desc[1]  # end
        state[2] = LayerState.LAYER_IDS[desc[2]]  # Layer id.
        state[7] = desc[3]  # memory footprint
        return state

    @staticmethod
    def relu_state(desc):
        state = [0] * LayerState.LAYER_STATE_LENGTH
        state[0] = desc[0]  # start
        state[1] = desc[1]  # end
        state[2] = LayerState.LAYER_IDS[desc[2]]  # Layer id.
        state[7] = desc[3]  # memory footprint
        return state

    @staticmethod
    def pool_state(desc):
        state = [0] * LayerState.LAYER_STATE_LENGTH
        state[0] = desc[0]  # start
        state[1] = desc[1]  # end
        state[2] = LayerState.LAYER_IDS[desc[2]]  # Layer id.
        state[3] = desc[3]  # Pool size
        state[4] = desc[4]  # Strides
        state[5] = desc[5]  # padding
        state[7] = desc[6]  # memory footprint
        return state
