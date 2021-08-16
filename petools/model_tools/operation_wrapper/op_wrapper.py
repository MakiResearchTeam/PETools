from typing import List, Tuple

from petools.tools.estimate_tools import Human


class OpWrapper:
    """
    This object performs a given Op on input humans. Each human will be provided with an individual Op instance.
    When instantiating an Op for a given human, it is being bind to the human's id (and saved to an internal register
    of ops) so that the Op instance can be reused when a human with the same id is encountered.
    """
    def __init__(self, op_init_fn):
        """
        Parameters
        ----------
        op_init_fn : func
            Function that create class which apply some operation on list of Human classes
            Class must have method with next signature:
                def __call__(self, human: Human) -> Human:
                    pass
            Class must return list of modified (or whatever) list of Human classes

        """
        self.op_init_fn = op_init_fn
        self.register = {}

    def __call__(self, humans: List[Human], **op_kwargs) -> List[Tuple[Human, object]]:
        """
        Returns results of the given Op applied to `humans`.

        Parameters
        ----------
        humans : List[Human]
            List of humans to be processed.
        op_kwargs : dict
            Supplement key-word arguments for the Op instances.

        Returns
        -------
        List[Tuple[Human, object]]
            A list of tuples (Human, operation result).
        """
        op_results = []
        for human in humans:
            if human.id == -1:
                # An exceptional case which is not being processed by the OpWrapper.
                # Humans with id=-1 are assumed to be erroneous.
                op_results.append((Human, None))
                continue
            op = self.register.get(str(human.id))
            if op is None:
                # Human with a new id has been encountered. Create a new instance of the Op
                op = self.op_init_fn()
                self.register[str(human.id)] = op
            op_result = op(human, **op_kwargs)
            op_results.append((human, op_result))
        return op_results
