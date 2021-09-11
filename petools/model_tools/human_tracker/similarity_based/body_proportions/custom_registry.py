from ..common import ExpAvgRegistry
from .representation import CustomRepresentation
from ..core import REPRESENTATION_ID


class CustomRegistry(ExpAvgRegistry):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_representation(self, id: REPRESENTATION_ID, representation: CustomRepresentation):
        super().update_representation(id, representation)
        if representation is None:
            return
        reg_repr = self.get_representation(id).representation
        reg_repr.xy = representation.xy
    
    def update_state(self):
        super(CustomRegistry, self).update_state()
        for id, holder in self.registry.items():
            xy_weights = holder.representation.xy_weights
            xy_weights.step()
            if self.debug_enabled:
                self.debug_log(f'Updated xy weights in representation with id={id}.')
                self.debug_log(f'decay={xy_weights.decay}, weights={xy_weights.weights}')
