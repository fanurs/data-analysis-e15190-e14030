import functools
import os
import pickle

def persistent_cache(cache_filepath: str):
    cache_filepath = os.path.expandvars(cache_filepath)
    def decorator_cache(func):
        cache = dict()
        if os.path.exists(cache_filepath):
            with open(cache_filepath, 'rb') as file:
                cache = pickle.load(file)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                return cache[key]

            result = func(*args, **kwargs)
            cache[key] = result
            with open(cache_filepath, 'wb') as file:
                pickle.dump(cache, file)
            return result

        return wrapper
    return decorator_cache
