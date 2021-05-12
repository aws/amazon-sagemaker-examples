# Logger class definition
import logging


class Logger(object):
    """
    Logger class for all DeepRacer Simulation Application logging
    """

    def __init__(self, logger_name, log_level=logging.INFO):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)

        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        self.logger.addHandler(handler)

    def get_logger(self):
        """
        Returns the logger object with all the required log settings.
        """
        return self.logger
