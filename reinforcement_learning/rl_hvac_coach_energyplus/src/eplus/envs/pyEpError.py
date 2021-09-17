class pyEpError(Exception):
    """Base class for pyEp Errors"""

    def __init__(self, message):
        super(pyEpError, self).__init__(message)


class VersionError(pyEpError):
    """Error Thrown when E+ Communications protocol is not 2."""

    def __init__(self, message=None):
        if message is None:
            message = "Incorrect Version of EnergyPlus communications protocol. Make sure your version of EnergyPlus supports version 2"
        super(VersionError, self).__init__(str(message))
        self.version = message


class EpWriteError(pyEpError):
    """Error thrown when appempting to write to a closed E+ instance"""

    def __init__(self, message=None):
        if message is None:
            message = "Error attempting to write to closed socket by EnergyPlus. Perhaps the simulation already finished?"
        super(EpWriteError, self).__init__(message)
        self.message = message


class EpReadError(pyEpError):
    """Error thrown when appempting to read from a closed E+ instance"""

    def __init__(self, message=None):
        if message is None:
            message = "Error attempting to read from closed EnergyPlus socket. Perhaps the simulation is already finished?"
        super(EpReadError, self).__init__(message)
        self.message = message


class MissingEpPathError(pyEpError):
    """Error thrown when the path to EnergyPlus is not specified."""

    def __init__(self, message=None):
        if message is None:
            message = "EnergyPlus path not specified. Set the default path with set_eplus_dir()"
        super(MissingEpPathError, self).__init__(message)
        self.message = message
