import abc

from petools.tools.estimate_tools import Human


# A basic notion of an operation to be performed on a human
class Op:
    @abc.abstractmethod
    def __call__(self, human: Human, **kwargs) -> Human:
        pass
