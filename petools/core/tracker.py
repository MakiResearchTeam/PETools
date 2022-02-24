from typing import List
from abc import abstractmethod

from petools.tools import Human
from petools.tools import LoggedEntity


class Tracker(LoggedEntity):
    def __call__(self, humans: List[Human], **kwargs) -> List[Human]:
        pass

    @abstractmethod
    def reset(self):
        pass

