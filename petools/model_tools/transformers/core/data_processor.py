import abc


class DataProcessor:
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass
