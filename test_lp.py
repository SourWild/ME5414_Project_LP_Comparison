import numpy as np
from main import generate_random_lp

def test_generation():
    c, A, b = generate_random_lp(10, 5)
    assert A.shape == (5, 10)
    assert len(c) == 10
    assert len(b) == 5
    print("Test Generation Passed")

if __name__ == '__main__':
    test_generation()
