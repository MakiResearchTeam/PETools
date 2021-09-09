import logging

from .logger_fabric import Logging


class LoggedEntity:
    def __init__(self):
        self.logger = Logging.get_logger(self.__class__.__name__)

    def debug_log(self, msg):
        self.logger.debug(msg)

    @property
    def debug_enabled(self):
        return self.logger.isEnabledFor(logging.DEBUG)
