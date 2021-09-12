import logging


class Logging:
    loggers = {}

    @staticmethod
    def get_logger(class_name) -> logging.Logger:
        if Logging.loggers.get(class_name):
            return Logging.loggers.get(class_name)
        else:
            logger = logging.getLogger(class_name)
            logger.addHandler(logging.NullHandler())
            logger.setLevel(logging.INFO)
            Logging.loggers[class_name] = logger
            return logger

    @staticmethod
    def set_debug_level():
        for logger in Logging.loggers.values():
            logger.setLevel(logging.DEBUG)

    @staticmethod
    def set_level(level):
        for logger in Logging.loggers.values():
            logger.setLevel(level)

    @staticmethod
    def set_handler(handler):
        for logger in Logging.loggers.values():
            logger.addHandler(handler)


def log(method):
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'logger'):
            self.logger.debug(f'Called {method}. args={args}, kwargs={kwargs}.')
        return method(self, *args, **kwargs)
    return wrapper
