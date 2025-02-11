import time
import json
import warnings

import numpy as np

from pathlib import Path

class NpEncoder(json.JSONEncoder):
    """
     Custom JSON encoder for NumPy data types.

     This encoder handles the serialization of NumPy data types to ensure they can be converted to JSON format.
     Specifically, it converts NumPy integers to Python integers, NumPy floating-point numbers to Python floats,
     and NumPy arrays to Python lists.

     I use this to serialize NumPy data types to JSON format, which is useful for platting graph using the D3.js package.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# A decorator to throw warning when we use deprecated methods/functions/routines
def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func

# A decorator to time the execution of a function
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper


def is_docker():
    cgroup = Path('/proc/self/cgroup')
    return Path('/.dockerenv').is_file() or cgroup.is_file() and 'docker' in cgroup.read_text()