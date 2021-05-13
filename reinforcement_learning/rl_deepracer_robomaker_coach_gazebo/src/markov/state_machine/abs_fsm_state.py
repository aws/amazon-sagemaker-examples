"""this module implement fsm state abs class"""

import abc

# Python 2 and 3 compatible Abstract class
ABC = abc.ABCMeta("ABC", (object,), {})


class AbsFSMState(ABC):
    """
    Abstract FSM state class
    """

    def __repr__(self):
        """
        Leverages the __str__ method to describe the State.
        """
        return self.__str__()

    def __str__(self):
        """
        Returns the name of the State.
        """
        return self.__class__.__name__

    def execute(self, input_val=None):
        """execute based on input from specific state

        Args:
            input_val (object): input values based on specific state machine
            for execution

        Returns:
            tuple: tuple with first value as state and other values are info
        """
        ret_val = self._execute(input_val)
        if ret_val is None:
            return None, None
        elif isinstance(ret_val, AbsFSMState):
            return ret_val, None
        elif isinstance(ret_val, tuple):
            if ret_val[0] is None:
                return None, ret_val[1:]
            elif isinstance(ret_val[0], AbsFSMState):
                return ret_val[0], ret_val[1:]
        raise ValueError(
            "Unexpected return type: {}. Expected return: (AbsFSMState, ...)".format(ret_val)
        )

    @abc.abstractmethod
    def _execute(self, input_val):
        """execute based on input from specific states

        Args:
            input_val (object): input values based on specific state machine
            for execution

        Raises:
            NotImplementedError: AbsFSMState _execute not implemented
        """
        raise NotImplementedError("AbsFSMState _execute not implemented")
