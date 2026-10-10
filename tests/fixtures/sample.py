"""Fixture for the parser smoke test. Line numbers are asserted in test_parser.py."""

import functools


def add(a, b):
    return a + b


def greet(name):
    message = f"hello {name}"
    return message


class Counter:
    def __init__(self):
        self.n = 0

    def bump(self):
        self.n += 1
        return self.n


@functools.cache
def square(x):
    return x * x
