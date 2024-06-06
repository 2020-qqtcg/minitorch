"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable
from numba import njit


# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """$f(x, y) = x * y$"""
    return x * y


def id(x: float) -> float:
    """$f(x) = x$"""
    return x


def add(x: float, y: float) -> float:
    """$f(x, y) = x + y$"""
    return x + y


def neg(x: float) -> float:
    """$f(x) = -x$"""
    return float(-x)  # force convert to float


def lt(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is less than y else 0.0"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """$f(x) =$ 1.0 if x is equal to y else 0.0"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """$f(x) =$ x if x is greater than y else y"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """$f(x) = |x - y| < 1e-2$"""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    r"""
    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$

    (See https://en.wikipedia.org/wiki/Sigmoid_function )

    Calculate as

    $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$

    for stability.
    """
    return (1.0 / (1.0 + math.exp(-x))) if x >= 0 else (math.exp(x) / (1.0 + math.exp(x)))


def relu(x: float) -> float:
    """
    $f(x) =$ x if x is greater than 0, else 0

    (See https://en.wikipedia.org/wiki/Rectifier_(neural_networks) .)
    """
    return x if x > 0 else 0


EPS = 1e-6


def log(x: float) -> float:
    """$f(x) = log(x)$"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """$f(x) = e^{x}$"""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """
    If $f = log$ as above, compute $d \times f'(x)$
    Args:
        x:
        d:

    Returns:
        f'(x)
    """
    if x == 0.0:
        raise RuntimeError("Zero cannot be used as a denominator.")
    return d * (1.0 / x)


def inv(x: float) -> float:
    """
    $f(x) = 1 / x$
    Args:
        x:

    Returns:
        1/x
    """
    if x == 0.0:
        raise RuntimeError("Zero cannot be used as a denominator.")
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    """
    If $f(x) = 1/x$ compute $d \times f'(x)$
    Args:
        x:
        d:

    Returns:

    """
    if x == 0.0:
        raise RuntimeError("Zero cannot be used as a denominator.")
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """
    If $f = relu$ compute $d \times f'(x)$
    relu has no derivative at zero, so we assume the derivative is zero.
    Args:
        x:
        d:

    Returns:

    """
    return d if x > 0 else 0.0


def sigmoid_back(x: float, d: float) -> float:
    """
    If $f = sigmod$ compute $d \times f'(x)$'
    Args:
        x:
        d:

    Returns:

    """
    return d * (1 - sigmoid(x)) * sigmoid(x)


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: Function from one value to one value.

    Returns:
         A function that takes a list, applies `fn` to each element, and returns a
         new list
    """

    def apply(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return apply


def negList(ls: Iterable[float]) -> Iterable[float]:
    """
    Use `map` and `neg` to negate each element in `ls`
    Args:
        ls: list of floats

    Returns:
        a list with negated elements
    """
    neg_list_func = map(neg)
    return neg_list_func(ls)


def zipWith(
        fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
        fn: combine two values

    Returns:
         Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """

    def apply(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x1, x2 in zip(ls1, ls2):
            ret.append(fn(x1, x2))
        return ret

    return apply


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Add the elements of `ls1` and `ls2` using `zipWith` and `add`

    Args:
        ls1: list of floats
        ls2: list of floats

    Returns:
        a list with added elements

    """
    add_list_func = zipWith(add)
    return add_list_func(ls1, ls2)


def reduce(
        fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""
    Higher-order reduce.

    Args:
        fn: combine two values
        start: start value $x_0$

    Returns:
         Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`
    """

    def apply(ls: Iterable[float]) -> float:
        ret = start
        for index, value in enumerate(ls[1:]):
            ret = fn(ret, value)
        return ret

    return apply


def sum(ls: Iterable[float]) -> float:
    """
    Sum up a list using `reduce` and `add`.

    Args:
        ls: a list of floats

    Returns:
        a sum of all elements
    """
    if len(list(ls)) == 0:
        return 0.0
    sum_func = reduce(add, next(iter(ls)))
    return sum_func(ls)


def prod(ls: Iterable[float]) -> float:
    """
    Product of a list using `reduce` and `mul`.

    Args:
        ls: a list of floats

    Returns:
        a product of all elements
    """
    prod_func = reduce(mul, next(iter(ls)))
    return prod_func(ls)
