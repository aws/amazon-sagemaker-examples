'''
Decorators are not really values, however we need some wrappers to improve
docstrings and other things around decorators.
'''

from jedi.inference.base_value import ValueWrapper, ValueSet


class Decoratee(ValueWrapper):
    def __init__(self, wrapped_value, original_value):
        super(Decoratee, self).__init__(wrapped_value)
        self._original_value = original_value

    def py__doc__(self):
        return self._original_value.py__doc__()

    def py__get__(self, instance, class_value):
        return ValueSet(
            Decoratee(v, self._original_value)
            for v in self._wrapped_value.py__get__(instance, class_value)
        )
