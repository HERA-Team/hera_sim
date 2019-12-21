# TODO: write module docstring
from abc import ABCMeta, abstractmethod
from .defaults import defaults

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
            cls._update_call_docstring()
            cls.is_multiplicative = is_multiplicative
            if not is_abstract:
                cls.__base__._models[cls.__name__] = cls

        def _extract_kwarg_values(self, **kwargs):
            # retrieve the default set of kwargs
            use_kwargs = self.kwargs.copy()

            # apply new defaults if the defaults class is active
            if defaults._override_defaults:
                kwargs = defaults.apply(use_kwargs, **kwargs)
            
            # make sure that any kwargs passed make it through
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

        def _update_call_docstring(cls):
            init_docstring = cls.__init__.__doc__
            call_docstring = cls.__call__.__doc__
            if any(
                ["Parameters" not in doc 
                 for doc in (init_docstring, call_docstring)
                ]
            ):
                return
            init_params = cls._extract_param_section(init_docstring)
            call_params = cls._extract_param_section(call_docstring)
            full_params = call_params + init_params
            cls.__call__.__doc__.replace(call_params, full_params)

        @staticmethod
        def _extract_param_section(docstring):
            # make a regular expression to capture section headings
            pattern = re.compile("[A-Za-z]+\n.*-+\n")
            # get the section headings
            section_headings = pattern.findall(docstring)
            if not section_headings[0].lower().startswith("param"):
                # TODO: make this error message a bit better
                # or just make it a warning instead
                raise SyntaxError(
                    "Please ensure that the 'Parameters' section of "
                    "the docstring comes first."
                )
            # get everything after the first heading
            param_section = docstring.partition(section_headings[0])[-1]
            # just return this if there are no more sections
            if len(section_headings) == 1:
                return param_section
            # return everything before the next section
            return param_section.partition(section_headings[1])[0]

    NewClass.__name__ = cls.__name__
    return NewClass

def list_discoverable_components():
    for cls in SimulationComponent.__subclasses__():
        for name, model in cls._models.items():
            name = ".".join([model.__module__, name])
            print(name)

