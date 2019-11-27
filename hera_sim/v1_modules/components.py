"""A module for keeping track of simulation components."""
from abc import ABC, abstractmethod
from cached_property import cached_property

class SimulationComponent(ABC, type):
    """Base class for constructing simulation components.

    """
    def __new__(cls, name, superclasses, attrs):
        # initialize a _models dictionary
        print("Calling SimulationComponent constructor.")
        cls._models = getattr(cls, "_models", {})
        return type.__new__(cls, name, superclasses, attrs)

class ComponentBase(metaclass=SimulationComponent):
    """Initialize a new class of components and its _models dict.

    """
    def __init_subclass__(cls, is_abstract=False, **kwargs):
        print("Calling ComponentBase __init_subclass__ routine.")
        print("is abstract is: {}".format(is_abstract))
        print("kwargs are: {}".format(kwargs))
        super().__init_subclass__(**kwargs)
        if not is_abstract:
            cls._models[cls.__name__] = cls._models.get(cls.__name__, [])
            cls._models[cls.__name__] = cls

    def __init__(self, **kwargs):
        print("calling ComponentBase initializor.")
        self.__name__ = kwargs.pop("__name__", "")
        self.kwargs = kwargs

    @cached_property
    def __alias__(self):
        """Get the handle for a class instance."""
        try:
            # check all the object ids in global scope, return
            # name of most recent occurrence of object id
            return [name for name, oid in globals().items()
                         if id(oid)==id(self)][0]
        except IndexError:
            return ''

    def __repr__(self):
        # maybe make this more informative?
        return self.__alias__

    @abstractmethod
    def __call__(self, **kwargs):
        # just check the kwargs and return their values
        if any([key not in self.kwargs.keys() for key in kwargs.keys()]):
            error_msg = "The following kwargs are not allowed: "
            for key in kwargs.keys():
                if key not in self.kwargs.keys():
                    error_msg += key + ", "
            raise ValueError(error_msg + ".")
        use_kwargs = self.kwargs.copy()
        use_kwargs.update(kwargs)
        return tuple(kwargs.values())

