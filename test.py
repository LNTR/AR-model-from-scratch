import numpy as np

foo = []


def foo_adder(bar):
    foo.append(4)
    print(foo)
    print(bar)


bar = np.array([1, 2, 3, 4, 5])
foo_adder_vec = np.vectorize(foo_adder)
foo_adder_vec(bar)
