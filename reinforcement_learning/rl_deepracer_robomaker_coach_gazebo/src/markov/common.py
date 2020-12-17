'''This module should be used for common classes and methods that can
   be used in any python module.
'''
import abc
from typing import Any

class ObserverInterface(metaclass=abc.ABCMeta):
    '''This class defines the interface for an observer, which can be registered
       to a sink and can receive notifications.
    '''
    def update(self, data: Any) -> None:
        '''Updates the observer with the data sent from the sink
           data - Data received from the sink
        '''
        raise NotImplementedError('Observer must implement update method')
