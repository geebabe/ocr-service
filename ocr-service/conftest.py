import sys
import os

# This file is loaded by pytest BEFORE any test modules are imported.
# We insert the service root directory at the front of sys.path to guarantee
# that the local project `app` package takes priority over any system-installed
# packages with the same name (e.g., a Flask `app` from Anaconda).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
