from __future__ import print_function
import pytest
from instant import inline

def test_inline():
    add_func = inline("double add(double a, double b){ return a+b; }",
                      cache_dir="test_cache")
    assert add_func(3, 4.5) == 7.5
