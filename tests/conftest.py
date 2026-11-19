"""Suite setup: this repository first on ``sys.path`` (the installed ``path_planning`` may be an older copy),
and one math thread unless the caller set otherwise (the default thread pool oversubscribes the small
linear-algebra calls of the tests and makes the run several times slower)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
