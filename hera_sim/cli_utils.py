"""Useful helper functions and argparsers for running simulations via CLI."""
from __future__ import annotations

import itertools
import math
import numpy as np
import os
import tracemalloc as tr
import warnings
from datetime import datetime
from pyuvdata import UVData
from rich._log_render import FormatTimeCallable
from rich.console import Console, ConsoleRenderable
from rich.containers import Renderables
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text, TextType
from string import Template
from typing import Literal
from collections.abc import Iterable

from .defaults import SEASON_CONFIGS
from .simulate import Simulator


def get_filing_params(config: dict):
    """Extract filing parameters from a configuration dictionary.

    Parameters
    ----------
    config
        The full configuration dict.

    Returns
    -------
    dict
        Filing parameter from the config, with default entries
        filled in.

    Raises
    ------
    ValueError
        If ``output_format`` not in "miriad", "uvfits", or "uvh5".
    """
    filing_params = dict(
        outdir=os.getcwd(),
        outfile_name="hera_sim_simulation.uvh5",
        output_format="uvh5",
        clobber=False,
    )
    filing_params.update(config.get("filing", {}))
    if filing_params["output_format"] not in ("miriad", "uvfits", "uvh5"):
        raise ValueError(
            "Output format not supported. Please use miriad, uvfits, or uvh5."
        )

    return filing_params


def validate_config(config: dict):
    """Validate the contents of a loaded configuration file.

    Parameters
    ----------
    config
        The full configuration dict.

    Raises
    ------
    ValueError
        If either insufficient information is provided, or the info
        is not valid.
    """
    if config.get("defaults") is not None:
        if type(config["defaults"]) is not str:
            raise ValueError(
                "Defaults in the CLI may only be specified using a string. "
                "The string used may specify either a path to a configuration "
                "yaml or one of the named default configurations."
            )

        if config["defaults"] in SEASON_CONFIGS.keys():
            return
        else:
            raise ValueError("Default configuration string not recognized.")

    freq_params = config.get("freq", {})
    time_params = config.get("time", {})
    array_params = config.get("telescope", {}).get("array_layout", {})
    if {} in (freq_params, time_params, array_params):
        raise ValueError("Insufficient information for initializing simulation.")

    freqs_ok = _validate_freq_params(freq_params)
    times_ok = _validate_time_params(time_params)
    array_ok = _validate_array_params(array_params)
    if not all([freqs_ok, times_ok, array_ok]):
        raise ValueError("Insufficient information for initializing simulation.")


def write_calfits(
    gains,
    filename,
    sim=None,
    freqs=None,
    times=None,
    x_orientation="north",
    clobber=False,
):
    """
    Write gains to disk as a calfits file.

    Parameters
    ----------
    gains: dict
        Dictionary mapping antenna numbers or (ant, pol) tuples to gains. Gains
        may either be spectra or waterfalls.
    filename: str
        Name of file, including desired extension.
    sim: :class:`pyuvdata.UVData` instance or :class:`~.simulate.Simulator` instance
        Object containing metadata pertaining to the gains to be saved. Does not
        need to be provided if both ``freqs`` and ``times`` are provided.
    freqs: array-like of float
        Frequencies corresponding to gains, in Hz. Does not need to be provided
        if ``sim`` is provided.
    times: array-like of float
        Times corresponding to gains, in JD. Does not need to be provided if
        ``sim`` is provided.
    x_orientation: str, optional
        Cardinal direction that the x-direction corresponds to. Defaults to the
        HERA configuration of north.
    clobber: bool, optional
        Whether to overwrite existing file in the case of a name conflict.
        Default is to *not* overwrite conflicting files.
    """
    from hera_cal.io import write_cal

    gains = gains.copy()

    if sim is not None:
        if not isinstance(sim, (Simulator, UVData)):
            raise TypeError("sim must be a Simulator or UVData object.")

        if isinstance(sim, Simulator):
            freqs = sim.freqs * 1e9
            times = sim.times
            sim_x_orientation = sim.data.x_orientation
        else:
            freqs = np.unique(sim.freq_array)
            times = np.unique(sim.time_array)
            sim_x_orientation = sim.x_orientation
        if sim_x_orientation is None:
            warnings.warn(
                "x_orientation not specified in simulation object."
                "Assuming that the x-direction points north.",
                stacklevel=1,
            )
        else:
            x_orientation = sim_x_orientation
    else:
        if freqs is None or times is None:
            raise ValueError(
                "If a simulation is not provided, then both frequencies and "
                "times must be specified."
            )

    # Update gain keys to conform to write_cal assumptions.
    # New Simulator gains have keys (ant, pol), so shouldn't need
    # special pre-processing.
    if all(np.issctype(type(ant)) for ant in gains.keys()):
        # Old-style, single polarization assumption.
        gains = {(ant, "Jee"): gain for ant, gain in gains.items()}

    # At the time of writing, the write_cal function *fails silently* if the
    # keys for the gain dictionary are not tuples specifying the antenna and
    # Jones polarization string. Using linear polarizations, as in (1, 'x'),
    # will cause the function to think that the gains do not exist, and so
    # will write a UVCal object whose gain_array consists solely of ones. In
    # order to prevent this behavior, it is necessary to ensure that the
    # keys of the gain dictionary are formatted correctly. This is also why
    # the x_orientation is *required* (and a value is assumed if none is
    # specified in the simulation object)--the Jones polarization strings
    # cannot be recovered from the usual linear polarization strings 'x', 'y'
    # without specifying the x-orientation.
    gains = _format_gain_dict(gains, x_orientation=x_orientation)
    # Ensure that all of the gains have the right shape.
    for antpol, gain in gains.items():
        if gain.ndim == 1:
            gains[antpol] = np.outer(np.ones(times.size), gain)

    write_cal(filename, gains, freqs, times, overwrite=clobber, return_uvc=False)


def _format_gain_dict(gains, x_orientation):
    """
    Format a gain dictionary to match the expectation from hera_cal.

    Parameters
    ----------
    gains: dict
        Dictionary mapping (ant, pol) tuples to gain spectra/waterfalls.
    x_orientation: str
        Cardinal direction corresponding to the array's x-direction.

    Returns
    -------
    gains: dict
        Dictionary mapping (ant, jpol) tuples to gain spectra/waterfalls. The
        distinction here is that the polarizations are Jones polarization
        strings, whereas the input gains may have ordinary linear polarization
        strings.
    """
    from hera_cal.io import jnum2str, jstr2num

    pol_array = list({antpol[1] for antpol in gains})
    jones_array = [
        jnum2str(
            jstr2num(pol, x_orientation=x_orientation), x_orientation=x_orientation
        )
        for pol in pol_array
    ]
    mapping = dict(zip(pol_array, jones_array))
    return {(antpol[0], mapping[antpol[1]]): gain for antpol, gain in gains.items()}


def _validate_freq_params(freq_params):
    """Ensure frequency parameters specified are sufficient."""
    allowed_params = (
        "Nfreqs",
        "start_freq",
        "bandwidth",
        "freq_array",
        "channel_width",
    )
    allowed_combinations = [
        combo
        for combo in itertools.combinations(allowed_params, 3)
        if "start_freq" in combo and "freq_array" not in combo
    ] + [("freq_array",)]
    for combination in allowed_combinations:
        if all(freq_params.get(param, None) is not None for param in combination):
            return True

    # None of the minimum necessary combinations are satisfied if we get here
    return False


def _validate_time_params(time_params):
    """Ensure time parameters specified are sufficient."""
    allowed_params = ("Ntimes", "start_time", "integration_time", "time_array")
    if time_params.get("time_array", None) is not None:
        return True
    elif all(time_params.get(param, None) is not None for param in allowed_params[:-1]):
        # Technically, start_time doesn't need to be specified, since it has a
        # default setting in io.py, but that might not be set in stone.
        return True
    else:
        return False


def _validate_array_params(array_params):
    """Ensure array layout is OK."""
    if isinstance(array_params, dict):
        # Shallow check; make sure each antenna position is a 3-vector.
        if all(len(pos) == 3 for pos in array_params.values()):
            return True
    elif isinstance(array_params, str):
        # Shallow check; just make sure the file exists.
        return os.path.exists(array_params)
    else:
        raise TypeError("Array layout must be a dictionary or path to a layout csv.")


class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    days = tdelta.days
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d = {
        "D": f"{days:02d}",
        "H": f"{hours + 24*days:02d}",
        "h": f"{hours:02d}",
        "M": f"{minutes:02d}",
        "S": f"{seconds:02d}",
    }

    t = DeltaTemplate(fmt)
    return t.substitute(**d)


def fmt_bytes(x: int):
    """Format a number in bytes."""
    order = int(math.log(x, 1024))
    x /= 1024**order

    if x >= 100.0:
        order += 1
        x /= 1024

    unit = [" B", "KB", "MB", "GB", "TB"][order]
    return f"{x:06.3f} {unit}"


class LogRender:
    def __init__(
        self,
        show_time: bool = True,
        show_level: bool = False,
        show_path: bool = True,
        time_format: str | FormatTimeCallable = "[%x %X]",
        omit_repeated_times: bool = True,
        level_width: int | None = 8,
        show_mem_usage: bool = True,
        mem_backend: Literal["tracemalloc", "psutil"] = "tracemalloc",
        show_time_as_diff: bool = False,
        delta_time_format: str = "%H:%M:%S",
    ) -> None:
        self.show_time = show_time
        self.show_level = show_level
        self.show_path = show_path
        self.time_format = time_format
        self.omit_repeated_times = omit_repeated_times
        self.level_width = level_width
        self._last_time: Text | None = None
        self._first_time: datetime | None = None
        self.delta_time_format = delta_time_format

        self.show_mem_usage = show_mem_usage
        self.mem_backend = mem_backend
        if mem_backend == "tracemalloc":
            if not tr.is_tracing():
                tr.start()

        elif mem_backend == "psutil":
            import psutil

            self._pr = psutil.Process
        else:
            raise ValueError(f"Invalid memory backend: {mem_backend}")

        self.show_time_as_diff = show_time_as_diff

    @classmethod
    def from_rich(
        cls,
        rich_log_render: LogRender,
        show_mem_usage: bool = True,
        mem_backend: Literal["tracemalloc", "psutil"] = "tracemalloc",
        show_time_as_diff: bool = False,
        delta_time_format: str = "%H:%M:%S",
    ):
        """
        Create a RichLog instance from a RichLog instance.

        :param rich_log_render:
            A RichLog instance.
        :param show_mem_usage:
            Whether to show
        """
        return cls(
            show_time=rich_log_render.show_time,
            show_level=rich_log_render.show_level,
            show_path=rich_log_render.show_path,
            time_format=rich_log_render.time_format,
            omit_repeated_times=rich_log_render.omit_repeated_times,
            level_width=rich_log_render.level_width,
            show_mem_usage=show_mem_usage,
            mem_backend=mem_backend,
            show_time_as_diff=show_time_as_diff,
            delta_time_format=delta_time_format,
        )

    def __call__(
        self,
        console: Console,
        renderables: Iterable[ConsoleRenderable],
        log_time: datetime | None = None,
        time_format: str | FormatTimeCallable | None = None,
        level: TextType = "",
        path: str | None = None,
        line_no: int | None = None,
        link_path: str | None = None,
    ) -> Table:
        output = Table.grid(padding=(0, 1))
        output.expand = True
        if self.show_time:
            output.add_column(style="log.time")
        if self.show_level:
            output.add_column(style="log.level", width=self.level_width)

        if self.show_mem_usage:
            output.add_column()

        output.add_column(ratio=1, style="log.message", overflow="fold")

        if self.show_path and path:
            output.add_column(style="log.path")

        row = []
        if self.show_time:
            log_time = log_time or console.get_datetime()
            if self._first_time is None:
                self._first_time = log_time

            if not self.show_time_as_diff:
                time_format = time_format or self.time_format
                if callable(time_format):
                    log_time_display = time_format(log_time)
                else:
                    log_time_display = Text(log_time.strftime(time_format))
                if log_time_display == self._last_time and self.omit_repeated_times:
                    row.append(Text(" " * len(log_time_display)))
                else:
                    row.append(log_time_display)
                    self._last_time = log_time_display
            else:
                time_diff = strfdelta(
                    log_time - self._first_time, self.delta_time_format
                )
                row.append(time_diff)

        if self.show_level:
            row.append(level)

        if self.show_mem_usage:
            if self.mem_backend == "psutil":
                m = self._pr().memory_info().rss
                row.append(fmt_bytes(m))
            elif self.mem_backend == "tracemalloc":
                m, p = tr.get_traced_memory()
                row.append(f"{fmt_bytes(m)} | {fmt_bytes(p)}")

        row.append(Renderables(renderables))
        if self.show_path and path:
            path_text = Text()
            path_text.append(
                path, style=f"link file://{link_path}" if link_path else ""
            )
            if line_no:
                path_text.append(":")
                path_text.append(
                    f"{line_no}",
                    style=f"link file://{link_path}#{line_no}" if link_path else "",
                )
            row.append(path_text)

        output.add_row(*row)
        return output


class RicherHandler(RichHandler):
    def __init__(
        self,
        *args,
        show_mem_usage: bool = True,
        mem_backend: Literal["tracemalloc", "psutil"] = "tracemalloc",
        show_time_as_diff: bool = False,
        delta_time_format: str = "%H:%M:%S",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._log_render = LogRender.from_rich(
            self._log_render,
            show_mem_usage=show_mem_usage,
            mem_backend=mem_backend,
            show_time_as_diff=show_time_as_diff,
            delta_time_format=delta_time_format,
        )
