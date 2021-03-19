"""A module providing discoverability features for hera_sim.
"""
from __future__ import annotations
import re
from abc import abstractmethod, ABCMeta
from copy import deepcopy
from .defaults import defaults
from typing import Dict, Optional, Tuple
from types import new_class
import inflection
from collections import defaultdict

_available_components = {}


class SimulationComponent(metaclass=ABCMeta):
    """Base class for defining simulation component models.

    This class serves two main purposes:
        - Provide a simple interface for discovering simulation
          component models (see :meth:`~list_discoverable_components`").
        - Ensure that each subclass can create abstract methods.

    The :meth:`~registry`: class decorator provides a simple way of
    accomplishing the above, while also providing some useful extra
    features.

    Attributes
    ----------
    is_multiplicative
        Specifies whether the model ``cls`` is a multiplicative
        effect. This parameter lets the :class:`~Simulator`:
        class determine how to apply the effect simulated by
        ``cls``. Default setting is False (i.e. the model is
        assumed to be additive unless specified otherwise).
    """

    is_multiplicative: bool = False
    _alias: Tuple[str] = tuple()

    def __init_subclass__(cls, is_abstract: bool = False):
        """Provide some useful augmentations to subclasses.

        Parameters
        ----------
        is_abstract
            Specifies whether the subclass ``cls`` is an abstract
            class or not. Classes that are not abstract are
            registered in the ``_models`` dictionary. This
            is the feature that provides a neat interface for
            automatic discoverability of component models. Default
            behavior is to register the subclass.

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

        For an example of how to use the ``component`` decorator,
        please refer to the following tutorial notebook:

        < ADD APPROPRIATE LINK HERE >
        """
        super().__init_subclass__()
        cls._update_call_docstring()
        if not is_abstract:
            for name in cls.get_aliases():
                cls._models[name] = cls

    @classmethod
    def get_aliases(cls):
        return (
            cls.__name__,
            cls.__name__.lower(),
            inflection.underscore(cls.__name__),
        ) + cls._alias

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
        if any(key not in self.kwargs for key in kwargs):
            error_msg = "The following keywords are not supported: "
            error_msg += ", ".join(key for key in kwargs if key not in self.kwargs)
            raise ValueError(error_msg)

    @classmethod
    def _update_call_docstring(cls):
        init_docstring = str(cls.__init__.__doc__)
        call_docstring = str(cls.__call__.__doc__)
        if any("Parameters" not in doc for doc in (init_docstring, call_docstring)):
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

    @classmethod
    def get_models(cls, with_aliases=False) -> Dict[str, SimulationComponent]:
        if with_aliases:
            return deepcopy(cls._models)
        else:
            return {model.__name__: model for model in set(cls._models.values())}

    @classmethod
    def get_model(cls, mdl: str) -> SimulationComponent:
        return cls._models[mdl]


# class decorator for tracking subclasses
def component(cls):
    """Decorator to create a new specific Component that tracks its models."""
    cls._models = {}
    # This function creates a new class dynamically.
    # The idea is to create a new class that is essentially the input cls, but has a
    # new superclass -- the SimulationComponent. We pass the "is_abstract" keyword into
    # the __init_subclass__ so that any class directly decorated with "@component" is seen to
    # be an abstract class, not an actual model. Finally, the exec_body just adds all
    # the stuff from cls into the new class.
    cls = new_class(
        name=cls.__name__,
        bases=(SimulationComponent,),
        kwds={"is_abstract": True},
        exec_body=lambda namespace: namespace.update(dict(cls.__dict__)),
    )
    _available_components[cls.__name__] = cls
    return cls


def get_all_components(with_aliases=False) -> Dict[str, Dict[str, SimulationComponent]]:
    return {
        cmp_name: cmp.get_models(with_aliases)
        for cmp_name, cmp in _available_components.items()
    }


def get_models(cmp: str, with_aliases: bool = False) -> Dict[str, SimulationComponent]:
    return get_all_components(with_aliases)[cmp.__name__]


def get_all_models(with_aliases: bool = False) -> Dict[str, SimulationComponent]:
    all = get_all_components(with_aliases)
    out = {}
    for k, v in all.items():
        out.update(v)
    return out


def get_model(mdl: str, cmp: Optional[str] = None) -> SimulationComponent:
    if cmp:
        return get_models(cmp, with_aliases=True)[mdl]
    else:
        return get_all_models(with_aliases=True)[mdl]


def print_all_components(with_aliases: bool = True):
    cmps = get_all_components(with_aliases)

    out = ""
    for cmp, models in cmps.items():
        out += f"{cmp}:\n"

        model_to_name = defaultdict(lambda: [])
        for name, model in models.items():
            model_to_name[model].append(name)

        for model, names in model_to_name.items():
            out += "  " + " | ".join(names) + "\n"
    print(out)
