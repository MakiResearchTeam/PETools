from .. import HumanRepresentation
from ..common import ExpAvgRegistry
from .representation import FancyRepresentation
from ..core import REPRESENTATION_ID
from .gaussian_measure import GaussianMeasure


class FancyRegistry(ExpAvgRegistry):
    def __init__(self, alpha: float = 0.2, expiration_time: int = 70):
        super().__init__(alpha=alpha, expiration_time=expiration_time)
        self.measure = GaussianMeasure()

    def update_representation(self, id: REPRESENTATION_ID, representation: FancyRepresentation):
        super().update_representation(id, representation)
        if representation is None:
            return
        reg_repr = self.get_representation(id).representation
        reg_repr.xy = representation.xy

        reg_repr.height = reg_repr.height * (1.0 - self.alpha) + representation.height * self.alpha

    def merge_reprs(self, old_repr: FancyRepresentation, new_repr: FancyRepresentation):
        sim = self.measure(old_repr, new_repr)
        old_repr.features = old_repr.features * (1 - self.alpha * sim) + new_repr.features * self.alpha * sim
    
    def update_state(self):
        super(FancyRegistry, self).update_state()
        for id, holder in self.registry.items():
            xy_weights = holder.representation.xy_weights
            xy_weights.step()
            if self.debug_enabled:
                self.debug_log(f'Updated xy weights in representation with id={id}.')
                self.debug_log(f'decay={xy_weights.decay}, weights={xy_weights.weights}')
