from ..common import ExpAvgRegistry
from .representation import FancyRepresentation
from ..core import REPRESENTATION_ID


class FancyRegistry(ExpAvgRegistry):
    def __init__(self, alpha: float = 0.2, expiration_time: int = 70):
        super().__init__(alpha=alpha, expiration_time=expiration_time)

    def update_representation(self, id: REPRESENTATION_ID, representation: FancyRepresentation):
        super().update_representation(id, representation)
        if representation is None:
            if self.debug_enabled:
                self.debug_log(f'Turning on weight decay on representation with id={id} because no update value received.')
            self.get_representation(id).representation.xy_weights.start_decay()
            return
        reg_repr = self.get_representation(id).representation
        reg_repr.xy = representation.xy
    
    def update_state(self):
        super(FancyRegistry, self).update_state()
        for id, holder in self.registry.items():
            xy_weights = holder.representation.xy_weights
            xy_weights.step()
            if self.debug_enabled:
                self.debug_log(f'Updated xy weights in representation with id={id}.')
                self.debug_log(f'decay={xy_weights.decay}, weights={xy_weights.weights}')