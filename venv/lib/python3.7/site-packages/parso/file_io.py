import os
from parso._compatibility import FileNotFoundError


class FileIO(object):
    def __init__(self, path):
        self.path = path

    def read(self):  # Returns bytes/str
        # We would like to read unicode here, but we cannot, because we are not
        # sure if it is a valid unicode file. Therefore just read whatever is
        # here.
        with open(self.path, 'rb') as f:
            return f.read()

    def get_last_modified(self):
        """
        Returns float - timestamp or None, if path doesn't exist.
        """
        try:
            return os.path.getmtime(self.path)
        except OSError:
            # Might raise FileNotFoundError, OSError for Python 2
            return None

    def _touch(self):
        try:
            os.utime(self.path, None)
        except FileNotFoundError:
            try:
                file = open(self.path, 'a')
                file.close()
            except (OSError, IOError):  # TODO Maybe log this?
                return False
        return True

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.path)


class KnownContentFileIO(FileIO):
    def __init__(self, path, content):
        super(KnownContentFileIO, self).__init__(path)
        self._content = content

    def read(self):
        return self._content
