import pytest
from astropy.time import Time
from astropy.utils import iers


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
