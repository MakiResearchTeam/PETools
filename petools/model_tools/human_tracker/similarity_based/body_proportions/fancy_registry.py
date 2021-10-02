from ..common import ExpAvgRegistry
from .representation import FancyRepresentation
from .gaussian_measure import GaussianMeasure


class FancyRegistry(ExpAvgRegistry):
    def __init__(self, alpha: float = 0.4, expiration_time: int = 70):
        super().__init__(alpha=alpha, expiration_time=expiration_time)
        self.measure = GaussianMeasure()

    def merge_reprs(self, old_repr: FancyRepresentation, new_repr: FancyRepresentation):
        # This allows for a bit smarter update of features.
        # Sudden changes are usually cause by anomalies. If such change has happened,
        # the similarity value will be lower. In such cases it may be better to preserve the old
        # representation so we simply multiply the momentum (alpha) with the similarity value.
        # Lower the similarity value, the less the representation is being changed.
        sim = self.measure(old_repr, new_repr)
        old_repr.features = old_repr.features * (1 - self.alpha * sim) + new_repr.features * self.alpha * sim
        old_repr.height = old_repr.height * (1.0 - self.alpha) + new_repr.height * self.alpha
        if new_repr.xy[0] > 0.0:
            old_repr.xy = new_repr.xy
    
    def update_state(self):
        super(FancyRegistry, self).update_state()
        for id, holder in self.registry.items():
            xy_weights = holder.representation.xy_weights
            xy_weights.step()
            if self.debug_enabled:
                self.debug_log(f'Updated xy weights in representation with id={id}.')
                self.debug_log(f'decay={xy_weights.decay}, weights={xy_weights.weights}')
