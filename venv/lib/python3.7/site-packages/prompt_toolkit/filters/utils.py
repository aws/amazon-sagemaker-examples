from __future__ import unicode_literals

from .base import Always, Filter, Never

__all__ = [
    'to_filter',
    'is_true',
]

_always = Always()
_never = Never()


def to_filter(bool_or_filter):
    """
    Accept both booleans and Filters as input and
    turn it into a Filter.
    """
    if not isinstance(bool_or_filter, (bool, Filter)):
        raise TypeError('Expecting a bool or a Filter instance. Got %r' % bool_or_filter)

    return {
        True: _always,
        False: _never,
    }.get(bool_or_filter, bool_or_filter)


def is_true(value):
    """
    Test whether `value` is True. In case of a Filter, call it.

    :param value: Boolean or `Filter` instance.
    """
    return to_filter(value)()
