from abc import abstractmethod

from petools.tools import Human


class FeatureExtractor:
	"""
	Extract features from a Human instance that are used later to determine its ID.
	"""
	@abstractmethod
	def __call__(self, human: Human, **kwargs):
		pass


