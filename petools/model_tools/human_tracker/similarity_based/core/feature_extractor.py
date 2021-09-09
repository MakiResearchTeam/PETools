from abc import abstractmethod

from petools.tools import Human, LoggedEntity


class FeatureExtractor(LoggedEntity):
	"""
	Extract features from a Human instance that are used later to determine its ID.
	"""
	@abstractmethod
	def __call__(self, human: Human, **kwargs):
		pass

	def reset(self):
		pass


