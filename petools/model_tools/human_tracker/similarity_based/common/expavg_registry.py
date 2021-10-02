from dataclasses import dataclass
from typing import List, Tuple

from ..core import RepresentationRegistry, REPRESENTATION_ID, HumanRepresentation


@dataclass
class RepresentationHolder:
    representation: HumanRepresentation
    id: int
    # Number of frames the owner of this feature was absent (for example, human walked out of the frame)
    n_absent: int = 0


class ExpAvgRegistry(RepresentationRegistry):
    def __init__(self, alpha: float = 0.9, expiration_time: int = 10):
        """
        Parameters
        ----------
        alpha : float
            The features being updated using exponential averaging: f = f_old * (1 - alpha) + f_new * alpha.
        expiration_time : int
            How much frame a human must be absent to remove it from the registry.
        """
        super().__init__()
        assert 0.0 < alpha <= 1.0, f'Alpha must be in (0, 1], but received alpha={alpha}.'
        self.alpha = alpha
        self.expiration_time = expiration_time
        self.id_counter = 0

    def register_representation(self, representation) -> REPRESENTATION_ID:
        holder = RepresentationHolder(representation, self.id_counter)
        self._register_representation(self.id_counter, holder)
        self.id_counter += 1
        return holder.id

    def update_representation(self, id: REPRESENTATION_ID, representation: HumanRepresentation):
        holder = self.get_representation(id)
        if representation is None:
            if self.debug_enabled:
                self.debug_log(f'For a human with id={id} received None. Probably human walked out of frame '
                              f'or the estimated pose is too dissimilar from the last one.\n')
            holder.n_absent += 1
            return
        else:
            holder.n_absent = 0
        self.merge_reprs(holder.representation, representation)

    def merge_reprs(self, old_repr: HumanRepresentation, new_repr: HumanRepresentation):
        # Update feature values using exponential averaging
        old_repr.features = \
            old_repr.features * (1.0 - self.alpha) + new_repr.features * self.alpha

    # noinspection PyTypeChecker
    @property
    def representations(self) -> List[Tuple[int, HumanRepresentation]]:
        reprs = [(id, x.representation) for id, x in self.registry.items()]
        return reprs

    def update_state(self):
        # Remove expired feature holders
        for id, holder in list(self.registry.items()):
            if holder.n_absent >= self.expiration_time:
                self.registry.pop(id)
                if self.debug_enabled:
                    self.debug_log(f'Representation holder with id={id} has expired. Removing.')

        super().update_state()

    def reset(self):
        super(ExpAvgRegistry, self).reset()
        self.id_counter = 0

