# TODO: write module docstring
from abc import ABCMeta

class SimulationComponent(metaclass=ABCMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._models = {}

# class decorator for tracking subclasses
def registry(cls):
    class NewClass(cls, SimulationComponent):
        def __init_subclass__(cls, is_abstract=False, 
                              is_multiplicative=False, **kwargs):
            super().__init_subclass__(**kwargs)
            cls.is_multiplicative = is_multiplicative
            if not is_abstract:
                cls.__base__._models[cls.__name__] = cls
    return NewClass
