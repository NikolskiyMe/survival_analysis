import time

"""
>>> def test():
        print('ok')
        
>>> res = timer(test)

>>> a = res()
    ok
    
>>> print(a)  # test
    0.2111098365
"""


def timer(func):
    def wrapper(*args, **kwargs):
        start_ts = time.time()
        func(*args, **kwargs)
        end_ts = time.time()
        tm = end_ts - start_ts
        # print(f"Function {func.__name__} time is {end_ts - start_ts} sec")
        return tm

    return wrapper
