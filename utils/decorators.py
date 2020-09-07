import time


def timeit(function):
    """
    Function decorator to find the time elapsed by a function.
    To time any func, just import it and place it on a function definition.

    @timeit
    def function(*args):
        ----
        return something

    Parameters
    ----------
    function : function
        any function you want to time

    Returns
    -------
    timed_function : function
        wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        time_elapsed = time.time() - start_time
        print('Time Elapsed: {:.4f}s'.format(time_elapsed))
        return result
    return wrapper
