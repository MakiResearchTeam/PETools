import logging
from typing import Tuple, List
from abc import abstractmethod
from petools.tools import LoggedEntity
from collections import Iterator

from .typing import REPRESENTATION_ID
from .feature_extractor import HumanRepresentation


class RepresentationNotFound(Exception):
    def __init__(self, id, *args, **kwargs):
        super().__init__(f'Feature with id={id} was not found in the registry. '
                         f'Please call `register_features` first.')


class RepresentationAlreadyRegistered(Exception):
    def __init__(self, id, *args, **kwargs):
        super().__init__(f'Feature with id={id} is already in the registry. '
                         f'Please use another id.')


class RepresentationRegistry(LoggedEntity):
    """
    Represents a buffer that holds representations (feature vectors) for registered (known) humans.
    It encapsulates how features are being processed from frame to frame (if such processing is needed, for example,
    exponential averaging among frames).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._registry = {}

    @property
    def registry(self):
        return self._registry

    def get_representation(self, id: REPRESENTATION_ID):
        f = self._registry.get(id)
        if f is None and self.debug_enabled:
            self.debug_log(f'Representation with id={id} was not found.')
            self.debug_log(f'Current registry contains the following representations:')
            for id, representation in self._registry.items():
                self.debug_log(f'id={id} / representation={representation}\n')
        return f

    def _register_representation(self, id: REPRESENTATION_ID, representation):
        """
        This method must be used for adding new features to the registry.

        Parameters
        ----------
        id : int
        representation : object
        """
        f = self.get_representation(id)
        if f is not None:
            raise RepresentationAlreadyRegistered(id)
        self.registry[id] = representation

        if self.debug_enabled:
            self.debug_log(f'Registered new representation with id={id}: ')
            self.debug_log(f'{representation}\n')

    @abstractmethod
    def register_representation(self, representation) -> REPRESENTATION_ID:
        pass

    def _update_representation(self, id: REPRESENTATION_ID, representation):
        # Make sure the feature exists.
        f = self.get_representation(id)
        if f is None:
            raise RepresentationNotFound(id)

        self._registry[id] = representation
        if self.debug_enabled:
            self.debug_log(f'Updated representation with id={id}:')
            self.debug_log(f'old_representation={self._registry[id]}')
            self.debug_log(f'new_representation={representation}.\n')

    @abstractmethod
    def update_representation(self, id: REPRESENTATION_ID, representation):
        pass

    def has_representations(self) -> bool:
        return len(self._registry) > 0

    @property
    @abstractmethod
    def representations(self) -> List[Tuple[int, HumanRepresentation]]:
        pass

    def update_state(self):
        """
        Being called at the end of each frame.
        """
        if self.debug_enabled:
            self.debug_log('Update state of the representations registry.\n')

    def reset(self):
        self._registry.clear()

        if self.debug_enabled:
            self.debug_log('Reset state of the representations registry.\n')

