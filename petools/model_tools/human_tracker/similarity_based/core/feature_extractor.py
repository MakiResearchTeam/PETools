from abc import abstractmethod
from dataclasses import dataclass

from petools.tools import Human, LoggedEntity


@dataclass
class HumanRepresentation:
	features: object


class FeatureExtractor(LoggedEntity):
	"""
	Extract features from a Human instance that are used later to determine its ID.
	"""
	@abstractmethod
	def __call__(self, human: Human, **kwargs) -> HumanRepresentation:
		pass

	def reset(self):
		pass


