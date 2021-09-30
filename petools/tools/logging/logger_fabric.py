import logging


class Logging:
    loggers = {}
    level = logging.INFO
    handler = logging.NullHandler()

    @staticmethod
    def get_logger(class_name) -> logging.Logger:
        if Logging.loggers.get(class_name):
            return Logging.loggers.get(class_name)
        else:
            logger = logging.getLogger(class_name)
            logger.addHandler(Logging.handler)
            logger.setLevel(Logging.level)
            Logging.loggers[class_name] = logger
            return logger

    @staticmethod
    def set_debug_level():
        Logging.level = logging.DEBUG
        for logger in Logging.loggers.values():
            logger.setLevel(logging.DEBUG)

    @staticmethod
    def set_level(level):
        Logging.level = level
        for logger in Logging.loggers.values():
            logger.setLevel(level)

    @staticmethod
    def set_handler(handler):
        Logging.handler = handler
        for logger in Logging.loggers.values():
            logger.addHandler(handler)


def log(method):
    def wrapper(self, *args, **kwargs):
        if hasattr(self, 'logger'):
            self.logger.debug(f'Called {method}. args={args}, kwargs={kwargs}.')
        return method(self, *args, **kwargs)
    return wrapper
