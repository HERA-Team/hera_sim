"""
A module for choosing which HERA season to simulate.
"""
from . import h1c, h2c
SEASONS = {'h1c':h1c, 'h2c':h2c}
DEFAULT_SEASON = 'h1c'

def set_default(season):
    """
    Method for setting the default HERA observing season.

    Arg:
        season (string):
            string designating which observing season to set as default

    Returns:
        None
    """
    assert season in SEASONS.keys()
    global DEFAULT_SEASON
    DEFAULT_SEASON = season

def get_season(season=None):
    """
    Method for retrieving a pointer to the desired observing season
    directory.

    Arg:
        season (string, optional): default=DEFAULT_SEASON
            string designating which observing season to simulate

    Returns:
        SEASON (module):
            pointer to desired observing season module
    """
    if season is None:
        season = DEFAULT_SEASON
    return SEASONS[season]
