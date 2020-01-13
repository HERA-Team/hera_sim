"""A module providing discoverability features for hera_sim.
"""

import functools
import re
from abc import ABCMeta, abstractmethod

from .defaults import defaults

class SimulationComponent(metaclass=ABCMeta):
    """Base class for defining simulation component models.

    This class serves two main purposes:
        - Provide a simple interface for discovering simulation 
          component models (see :meth:`~list_discoverable_components`").
        - Ensure that each subclass can create abstract methods.

    The :meth:`~registry`: class decorator provides a simple way of 
    accomplishing the above, while also providing some useful extra
    features.
    """
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._models = {}

# class decorator for tracking subclasses
def registry(cls):
    class NewClass(cls, SimulationComponent):
        def __init_subclass__(cls, is_abstract=False, 
                              is_multiplicative=False, **kwargs):
            """Provide some useful augmentations to subclasses.

            Parameters
            ----------
            is_abstract : bool, optional
                Specifies whether the subclass ``cls`` is an abstract 
                class or not. Classes that are not abstract are 
                registered in the ``NewClass._models`` dictionary. This 
                is the feature that provides a neat interface for 
                automatic discoverability of component models. Default 
                behavior is to register the subclass.

            is_multiplicative : bool, optional
                Specifies whether the model ``cls`` is a multiplicative 
                effect. This parameter lets the :class:`~Simulator`: 
                class determine how to apply the effect simulated by 
                ``cls``. Default setting is False (i.e. the model is 
                assumed to be additive unless specified otherwise).

            **kwargs
                Passed to the superclass ``__init_subclass__`` method.
            
            Notes
            -----
            This subclass initialization routine also automatically 
            updates the ``__call__`` docstring for the subclass with 
            the parameters from the ``__init__`` docstring if both 
            methods are documented in the numpy style. This decision was 
            made because the convention for defining a new component 
            model is to have optional parameters be specified on class 
            instantiation, but with the option to override the 
            parameters when the class instance is called. In lieu of 
            repeating the optional parameters with their defaults, all 
            component model signatures consist of only required 
            parameters and variable keyword arguments.

            For an example of how to use the ``registry`` decorator, 
            please refer to the following tutorial notebook:
            
            < ADD APPROPRIATE LINK HERE >
            """
            super().__init_subclass__(**kwargs)
            cls._update_call_docstring()
            cls.is_multiplicative = is_multiplicative
            if not is_abstract:
                cls.__base__._models[cls.__name__] = cls

        def _extract_kwarg_values(self, **kwargs):
            """Return the (optionally updated) model's optional parameters.

            Parameters
            ----------
            **kwargs
                An unpacked dictionary of optional parameter values 
                appropriate for the model. These are received directly 
                from the subclass's ``__call__`` method.

            Returns
            -------
            use_kwargs : dict values
                Potentially updated parameter values for the parameters 
                passed in ``**kwargs``. This allows for a very simple 
                interface with the :module:`~defaults`: module, which 
                will automatically update parameter default values if 
                active.
            """
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

        @classmethod
        def _update_call_docstring(cls):
            init_docstring = str(cls.__init__.__doc__)
            call_docstring = str(cls.__call__.__doc__)
            if any(
                ["Parameters" not in doc 
                 for doc in (init_docstring, call_docstring)
                ]
            ):
                return
            init_params = cls._extract_param_section(init_docstring)
            call_params = cls._extract_param_section(call_docstring)
            full_params = call_params + init_params
            cls.__call__.__doc__ = call_docstring.replace(call_params, full_params)

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

