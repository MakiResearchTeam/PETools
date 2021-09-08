from petools.tools import Logging


class LoggedEntity:
    def __init__(self):
        self.logger = Logging.get_logger(self.__class__.__name__)
