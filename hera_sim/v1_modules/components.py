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

        def _extract_kwarg_values(self, **kwargs):
            use_kwargs = self.kwargs.copy()
            use_kwargs.update(kwargs)
            return use_kwargs.values()

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        @abstractmethod
        def __call__(self, **kwargs):
            self._check_kwargs(**kwargs)

        def _check_kwargs(self, **kwargs):
            if any([key not in self.kwargs for key in kwargs]):
                raise ValueError("The following keywords are not "
                                 "supported: "
                                 ", ".join([key for key in kwargs
                                            if key not in self.kwargs]))
    return NewClass
