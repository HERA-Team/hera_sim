"""A module providing discoverability features for hera_sim."""

from __future__ import annotations

import re
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy
from types import new_class

from .defaults import defaults

_available_components = {}


class SimulationComponent(metaclass=ABCMeta):
    """Base class for defining simulation component models.

    This class serves two main purposes:
        - Provide a simple interface for discovering simulation
          component models (see :meth:`~list_discoverable_components`).
        - Ensure that each subclass can create abstract methods.

    The :meth:`~component`: class decorator provides a simple way of
    accomplishing the above, while also providing some useful extra
    features.

    Attributes
    ----------
    is_multiplicative
        Specifies whether the model ``cls`` is a multiplicative
        effect. This parameter lets the :class:`~hera_sim.simulate.Simulator`
        class determine how to apply the effect simulated by
        ``cls``. Default setting is False (i.e. the model is
        assumed to be additive unless specified otherwise).
    return_type
        Whether the returned result is per-antenna, per-baseline, or the full
        data array. This tells the :class:`~hera_sim.simulate.Simulator` how
        it should handle the returned result.
    attrs_to_pull
        Dictionary mapping parameter names to :class:`~hera_sim.simulate.Simulator`
        attributes to be retrieved when setting up for simulation.
    """

    #: Whether this systematic multiplies existing visibilities
    is_multiplicative: bool = False
    #: Whether this systematic contains a randomized component
    is_randomized: bool = False
    #: Whether the returned value is per-antenna, per-baseline, or the full array
    return_type: str | None = None
    # This isn't exactly safe, but different instances of a class should
    # have the same call signature, so this should actually be OK.
    #: Mapping between parameter names and Simulator attributes
    attrs_to_pull: dict = {}

    _alias: tuple[str] = tuple()

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
    def get_aliases(cls) -> tuple[str]:
        """Get all the aliases by which this model can be identified."""
        return (cls.__name__.lower(),) + cls._alias

    def _extract_kwarg_values(self, **kwargs):
        """Return the (optionally updated) model's optional parameters.

        Parameters
        ----------------
        **kwargs
            Optional parameter values appropriate for the model. These are received
            directly from the subclass's ``__call__`` method.

        Returns
        -------
        use_kwargs : dict values
            Potentially updated parameter values for the parameters
            passed in. This allows for a very simple
            interface with the :mod:`~hera_sim.defaults`: module, which
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
        """Compute the component model."""
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
    def get_models(cls, with_aliases=False) -> dict[str, SimulationComponent]:
        """Get a dictionary of models associated with this component."""
        if with_aliases:
            return deepcopy(cls._models)
        else:
            return {
                model.__name__.lower(): model for model in set(cls._models.values())
            }

    @classmethod
    def get_model(cls, mdl: str) -> SimulationComponent:
        """Get a model with a particular name (including aliases)."""
        return cls._models[mdl.lower()]


# class decorator for tracking subclasses
def component(cls):
    """Decorator to create a new :class:`SimulationComponent` that tracks its models."""
    cls._models = {}
    # This function creates a new class dynamically.
    # The idea is to create a new class that is essentially the input cls, but has a
    # new superclass -- the SimulationComponent. We pass the "is_abstract" keyword into
    # the __init_subclass__ so that any class directly decorated with "@component" is
    # seen to be an abstract class, not an actual model. Finally, the exec_body just
    # adds all the stuff from cls into the new class.
    cls = new_class(
        name=cls.__name__,
        bases=(SimulationComponent,),
        kwds={"is_abstract": True},
        exec_body=lambda namespace: namespace.update(dict(cls.__dict__)),
    )
    _available_components[cls.__name__] = cls

    # Don't require users to write a class docstring (even if they should)
    if cls.__doc__ is None:
        cls.__doc__ = """"""

    # Add some common text to the docstring.
    cls.__doc__ += """

    This is an *abstract* class, and should not be directly instantiated. It represents
    a "component" -- a modular part of a simulation for which several models may be
    defined. Models for this component may be defined by subclassing this abstract base
    class and implementing (at least) the :meth:`__call__` method. Some of these are
    implemented within hera_sim already, but custom models may be implemented outside
    of hera_sim, and used on equal footing with the the internal models (as long as
    they subclass this abstract component).

    As with all components, all parameters that define the behaviour of the model are
    accepted at class instantiation. The :meth:`__call__` method actually computes the
    simulated effect of the component (typically, but not always, a set of visibilities
    or gains), by *default* using these parameters. However, these parameters can be
    over-ridden at call-time. Inputs such as the frequencies, times or baselines at
    which to compute the effect are specific to the call, and do not get passed at
    instantiation.
    """
    return cls


def get_all_components(with_aliases=False) -> dict[str, dict[str, SimulationComponent]]:
    """Get a dictionary of component names mapping to a dictionary of models."""
    return {
        cmp_name.lower(): cmp.get_models(with_aliases)
        for cmp_name, cmp in _available_components.items()
    }


def get_models(cmp: str, with_aliases: bool = False) -> dict[str, SimulationComponent]:
    """Get a dict of model names mapping to model classes for a particular component."""
    return get_all_components(with_aliases)[cmp.lower()]


def get_all_models(with_aliases: bool = False) -> dict[str, SimulationComponent]:
    """Get a dictionary of model names mapping to their classes for all possible models.

    See Also
    --------
    :func:`get_models`
        Return a similar dictionary but filtered to a single kind of component.
    """
    all_cmps = get_all_components(with_aliases)
    out = {}
    for models in all_cmps.values():
        # models here is a dictionary of all models of a particular component.
        out.update(models)
    return out


def get_model(mdl: str, cmp: str | None = None) -> type[SimulationComponent]:
    """Get a particular model, based on its name.

    Parameters
    ----------
    mdl
        The name (or alias) of the model to get.
    cmp
        If desired, limit the search to a specific component name. This helps if there
        are name clashes between models.

    Returns
    -------
    cmp
        The :class:`SimulationComponent` corresponding to the desired model.
    """
    if cmp:
        return get_models(cmp, with_aliases=True)[mdl.lower()]
    else:
        return get_all_models(with_aliases=True)[mdl.lower()]


def list_all_components(with_aliases: bool = True) -> str:
    """Lists all discoverable components.

    Parameters
    ----------
    with_aliases
        If True, also include model aliases in the output.

    Returns
    -------
    str
        A string summary of the available models.
    """
    cmps = get_all_components(with_aliases)

    out = ""
    for cmp, models in cmps.items():
        out += f"{cmp}:\n"

        model_to_name = defaultdict(list)
        for name, model in models.items():
            model_to_name[model].append(name)

        for names in model_to_name.values():
            out += "  " + " | ".join(names) + "\n"
    return out
