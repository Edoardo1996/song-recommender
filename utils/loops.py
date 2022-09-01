"""
Module handling loops and progress bars
"""
import time
from math import inf
from tqdm import tqdm

def _format_description(text, elapsed_time, remaining_time) -> str:
    return f"""Elapsed time: {elapsed_time}\n
               Remaining time: {remaining_time}\n
               {text}"""

def timed_loop(pbar, description = None, timeout=inf):
    """
    Loop that shows a progress bar along with elapsed and remaining time
    """
    start_time = time.time()
    iterator = iter(pbar)

    # Retrieve total number of iteration
    total_its = pbar.total

    while True:
        elapsed_time = time.time() - start_time
        pbar.set_description(f"ciao {elapsed_time}")
        
        if elapsed_time > timeout:
            raise TimeoutError("long_running_function took too long!")

        try:
            yield next(iterator)
        except StopIteration:
            pass


def long_running_function(n, timeout=inf):
    bar = tqdm(list(range(n)))
    for _ in timed_loop(bar, "ciao"):
        time.sleep(0.1)


long_running_function(20)