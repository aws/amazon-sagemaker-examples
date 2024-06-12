"""
To ensure compatibility from Python ``2.7`` - ``3.3``, a module has been
created. Clearly there is huge need to use conforming syntax.
"""
import os
import sys
import platform

# unicode function
try:
    unicode = unicode
except NameError:
    unicode = str

is_pypy = platform.python_implementation() == 'PyPy'


def use_metaclass(meta, *bases):
    """ Create a class with a metaclass. """
    if not bases:
        bases = (object,)
    return meta("HackClass", bases, {})


try:
    encoding = sys.stdout.encoding
    if encoding is None:
        encoding = 'utf-8'
except AttributeError:
    encoding = 'ascii'


def u(string):
    """Cast to unicode DAMMIT!
    Written because Python2 repr always implicitly casts to a string, so we
    have to cast back to a unicode (and we know that we always deal with valid
    unicode, because we check that in the beginning).
    """
    if sys.version_info.major >= 3:
        return str(string)

    if not isinstance(string, unicode):
        return unicode(str(string), 'UTF-8')
    return string


try:
    # Python 3.3+
    FileNotFoundError = FileNotFoundError
except NameError:
    # Python 2.7 (both IOError + OSError)
    FileNotFoundError = EnvironmentError
try:
    # Python 3.3+
    PermissionError = PermissionError
except NameError:
    # Python 2.7 (both IOError + OSError)
    PermissionError = EnvironmentError


def utf8_repr(func):
    """
    ``__repr__`` methods in Python 2 don't allow unicode objects to be
    returned. Therefore cast them to utf-8 bytes in this decorator.
    """
    def wrapper(self):
        result = func(self)
        if isinstance(result, unicode):
            return result.encode('utf-8')
        else:
            return result

    if sys.version_info.major >= 3:
        return func
    else:
        return wrapper


if sys.version_info < (3, 5):
    """
    A super-minimal shim around listdir that behave like
    scandir for the information we need.
    """
    class _DirEntry:

        def __init__(self, name, basepath):
            self.name = name
            self.basepath = basepath

        @property
        def path(self):
            return os.path.join(self.basepath, self.name)

        def stat(self):
            # won't follow symlinks
            return os.lstat(os.path.join(self.basepath, self.name))

    def scandir(dir):
        return [_DirEntry(name, dir) for name in os.listdir(dir)]
else:
    from os import scandir
