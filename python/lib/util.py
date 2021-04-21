import time

# Time the duration of the given lambda and print it.
def timeit(f, name=''):
    s = time.time()
    ret = f()
    e = time.time()
    print(f'Duration [{name}]: {e-s:4.4f}')
    return ret
