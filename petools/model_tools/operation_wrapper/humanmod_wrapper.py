from typing import List, Tuple

from .op_wrapper import OpWrapper
from petools.tools.estimate_tools import Human


class HumanModWrapper(OpWrapper):
    """
    A subclass that simply unpacks results of the Op.
    An op in the context of HumanModWrapper is an Op that modifies or creates a new human.
    """
    def __call__(self, humans: list, **op_kwargs) -> List[Human]:
        """
        Returns results of the given Op (human modifier) applied to `humans`.

        Parameters
        ----------
        humans : List[Human]
            List of humans to be processed.
        op_kwargs : dict
            Supplement key-word arguments for the Op instances.

        Returns
        -------
        List[Human]
            A list of tuples (Human, operation result).
        """
        # noinspection PyTypeChecker
        mod_results: List[Tuple[Human, Human]] = super(HumanModWrapper, self).__call__(humans, **op_kwargs)
        updated_humans = []
        for old_human, new_human in mod_results:
            if new_human is None:
                # Something made it impossible to modify that human.
                # Leave an old human as a result.
                updated_humans.append(old_human)
                # Skip id assignment
                continue
            # Restore `id`, because some modules can recreate human class (so id in human will be dropped)
            # In order to keep `id` through different modules, assign old `id` to the new instance
            new_human.id = old_human.id
            updated_humans.append(new_human)
        return updated_humans
