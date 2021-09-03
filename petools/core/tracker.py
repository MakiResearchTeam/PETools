from typing import List

from petools.tools import Human


class Tracker:
    def __call__(self, humans: List[Human], **kwargs) -> List[Human]:
        pass
