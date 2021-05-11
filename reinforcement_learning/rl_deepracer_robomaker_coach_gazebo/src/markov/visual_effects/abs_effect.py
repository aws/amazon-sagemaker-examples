import abc

from markov.visual_effects.effect_manager import EffectManager

# Python 2 and 3 compatible Abstract class
ABC = abc.ABCMeta("ABC", (object,), {})


class AbstractEffect(ABC):
    """
    Abstract Effect class

    Effect Call order:
    - attach -> [add to effect_manager] -> on_attach
    After effect is attached to effect manager, effect_manager will call update asynchronously
    - update -> (_lazy_init) -> _update
      - (_laze_init) is called only once at initial update call
    - detach -> [remove from effect_manager] -> on_detach
    """

    def __init__(self):
        self._is_init_called = False

    def attach(self):
        """
        Attach the effect to effect manager.
        """
        EffectManager.get_instance().add(self)

    def detach(self):
        """
        Detach the effect from effect manager.
        """
        EffectManager.get_instance().remove(self)

    def update(self, delta_time):
        """
        Update the effect

        Args:
            delta_time (float): the change of time in second from last call
        """
        if not self._is_init_called:
            self._is_init_called = True
            self._lazy_init()
        self._update(delta_time)

    def on_attach(self):
        """
        Subclass should override this if any action needed during attach.
        """
        pass

    def on_detach(self):
        """
        Subclass should override this if any action needed during detach.
        """
        pass

    def _lazy_init(self):
        """
        Subclass should override this to do lazy-initialize just before first update call.
        """
        pass

    @abc.abstractmethod
    def _update(self, delta_time):
        """
        Subclass must override this to update the effect.

        Args:
            delta_time (float): the change of time in second from last call
        """
        raise NotImplementedError("Effect must implement this function")
