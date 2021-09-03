from typing import Tuple
from abc import abstractmethod
from logging import getLogger

from .typing import FEATURE_ID


class FeatureNotFound(Exception):
    def __init__(self, id, *args, **kwargs):
        super().__init__(f'Feature with id={id} was not found in the registry. '
                         f'Please call `register_features` first.')


class FeatureAlreadyRegistered(Exception):
    def __init__(self, id, *args, **kwargs):
        super().__init__(f'Feature with id={id} is already in the registry. '
                         f'Please use another id.')


class FeatureIterator:
    """
    Iterates over already registered features.
    """

    @abstractmethod
    def __next__(self) -> Tuple[FEATURE_ID, object]:
        pass


class FeatureRegistry:
    """
    Represents a buffer that holds feature vectors for registered (known) humans.
    It encapsulates how features are being processed from frame to frame (if such processing is needed, for example,
    exponential averaging among frames).
    """

    def __init__(self, *args, **kwargs):
        self.logger = getLogger(self.__class__.__name__)
        init_info = 'Initialized FeatureRegistry: \n' \
                    f'class_name: {self.__class__.__name__}\n' \
                    f'args: {args}\n' \
                    f'kwargs: {kwargs}'
        self.logger.info(init_info)
        self._registry = {}

    @property
    def registry(self):
        return self._registry

    def get_feature(self, id: FEATURE_ID):
        f = self._registry.get(id)
        if f is None:
            debug_msg = 'Call _register_features: \n' \
                        'registry=' + str(self.registry) + '\n' \
                                                           f'id={id}\n'
            self.logger.debug(debug_msg)
            self.logger.error(f'Feature with id={id} was not found.')
        return f

    def _register_features(self, id: FEATURE_ID, features):
        """
        This method must be used for adding new features to the registry.

        Parameters
        ----------
        id : int
        features : object
        """
        f = self.get_feature(id)
        if f is not None:
            raise FeatureAlreadyRegistered(id)
        self.registry[id] = features
        self.logger.info(f'Registered new features with id={id}.')

    @abstractmethod
    def register_features(self, features) -> FEATURE_ID:
        pass

    def _update_features(self, id: FEATURE_ID, features):
        # Make sure the feature exists.
        f = self.get_feature(id)
        if f is None:
            raise FeatureNotFound(id)

        self._registry[id] = features
        self.logger.info(f'Updated features with id={id}.')

    @abstractmethod
    def update_features(self, id: FEATURE_ID, features):
        pass

    def has_features(self) -> bool:
        return len(self._registry) > 0

    @abstractmethod
    def __iter__(self) -> FeatureIterator:
        pass

    def update_state(self):
        """
        Being called at the end of each frame.
        """
        pass
