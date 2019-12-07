# TODO: write module docstring
from abc import ABCMeta, abstractmethod

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

        def _extract_kwarg_values(self, **kwargs):
            use_kwargs = self.kwargs.copy()
            use_kwargs.update(kwargs)
            return use_kwargs.values()

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @abstractmethod
        def __call__(self, **kwargs):
            pass

        def _check_kwargs(self, **kwargs):
            if any([key not in self.kwargs for key in kwargs]):
                error_msg = "The following keywords are not supported: "
                error_msg += ", ".join([key for key in kwargs 
                                        if key not in self.kwargs])
                raise ValueError(error_msg)

    NewClass.__name__ = cls.__name__
    return NewClass
