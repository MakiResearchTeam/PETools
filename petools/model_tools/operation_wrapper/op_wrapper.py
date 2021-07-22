

class OPWrapper:

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

    def __call__(self, humans: list, **op_kwargs) -> list:
        """
        Returns updated human list
        According to init operation

        """
        updated_humans = []
        for human in humans:
            # Todo: think about this case
            if human.id == -1:
                continue
            # Save id
            old_id = human.id
            # Take mod
            mod = self.register.get(str(human.id))
            # If not found (new human)
            if mod is None:
                # Init new
                mod = self.op_init_fn()
                self.register[str(human.id)] = mod
            # Apply mod on human
            updated_human = mod(human, **op_kwargs)
            # Restore id
            # Some modules recreate human class
            # In order to keep id through different modules, apply old id
            updated_human.id = old_id
            # Store updated human
            updated_humans.append(updated_human)

        return updated_humans
