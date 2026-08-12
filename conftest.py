"""Root conftest: put the project root on sys.path so tests can import modules."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
