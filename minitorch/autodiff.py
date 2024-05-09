from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    if len(vals) < arg + 1:
        raise RuntimeError("Not enough arguments")

    d_vals = list(vals[:])
    d_vals[arg] += epsilon

    return (f(*d_vals) - f(*vals)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """

    # we asume there has no circle, simplifying the sort

    topological_set = set()
    return topological_sort_dfs(variable, topological_set)


def topological_sort_dfs(variable: Variable, topological_set: set) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.
    Args:
        variable: a variable in topological order
        topological_set: record node viewed

    Returns:
        Non-constant Variables in topological order starting from the node.
    """
    topological_list = []
    for v in variable.parents:
        if v.unique_id not in topological_set:
            topological_set.add(v.unique_id)
            if not v.is_constant():
                topological_list += topological_sort_dfs(v, topological_set)
    topological_list.append(variable)
    return topological_list[::-1]


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """

    # get an ordered queue and generate a dict ( scalar : detivattive)
    variables = topological_sort(variable)
    scalar_dict = {key: None for key in variables}
    scalar_dict[variable] = deriv

    # backpropagate
    for v in variables:
        if v.is_leaf():
            continue
        if scalar_dict[v] is None:
            print(1)
        for cr in v.chain_rule(scalar_dict[v]):
            if scalar_dict[cr[0]] is not None:
                scalar_dict[cr[0]] += cr[1]
            else:
                scalar_dict[cr[0]] = cr[1]

    # accumulate derivative for leaf
    for v in variables:
        if v.is_leaf():
            v.accumulate_derivative(scalar_dict[v])


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
