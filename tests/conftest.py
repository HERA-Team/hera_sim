import importlib.util
import os
from pathlib import Path

# Restrict UCX to shared-memory/loopback transports before anything can
# initialise MPI (hera_sim.visibilities.cli calls MPI.Init() at import, so this
# happens in every xdist worker during collection). On runners exposing an
# RDMA-capable NIC (e.g. Azure's MANA), UCX fails to open the verbs transport
# and aborts the interpreter from C. These tests are single-rank, so no network
# transport is needed.
os.environ.setdefault("UCX_TLS", "self,sm,tcp")

import pytest
from astropy.time import Time
from astropy.utils import iers


def pytest_collection_modifyitems(config, items):
    """Skip tests marked with ``mpi`` if mpi4py isn't installed."""
    if importlib.util.find_spec("mpi4py") is not None:
        return

    skip_mpi = pytest.mark.skip(reason="mpi4py is not installed")
    for item in items:
        if "mpi" in item.keywords:
            item.add_marker(skip_mpi)


@pytest.fixture(autouse=True, scope="session")
def setup_and_teardown_package():
    # Try to download the latest IERS table. If the download succeeds, run a
    # computation that requires the values, so they are cached for all future
    # tests. If it fails, turn off auto downloading for the tests and turn it
    # back on once all tests are completed (done by extending auto_max_age).
    # Also, the checkWarnings function will ignore IERS-related warnings.
    try:
        t1 = Time.now()
        t1.ut1
    except Exception:
        iers.conf.auto_max_age = None

    yield

    iers.conf.auto_max_age = 30

@pytest.fixture(scope='session')
def repodir() -> Path:
    return Path(__file__).parent.parent

@pytest.fixture(scope='session')
def exampledir(repodir: Path) -> Path:
    return repodir / 'config_examples'
