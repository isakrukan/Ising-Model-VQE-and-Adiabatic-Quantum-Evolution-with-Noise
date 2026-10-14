"""
Shared pytest configuration.

``Code/`` is placed on ``sys.path`` through the ``pythonpath`` setting in
pyproject.toml, so the tests import the analysis modules exactly the way the
scripts do.
"""

import matplotlib

# The analysis modules import matplotlib.pyplot at module level; force the
# non-interactive backend so nothing tries to open a window.
matplotlib.use("Agg")
