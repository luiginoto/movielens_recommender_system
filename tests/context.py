import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules import als

'''Then, within the individual test modules, import the module like so:

from .context import sample

'''