from typing import List

from petools.tools import Human
from petools.tools import LoggedEntity


class Tracker(LoggedEntity):
    def __call__(self, humans: List[Human], **kwargs) -> List[Human]:
        pass

