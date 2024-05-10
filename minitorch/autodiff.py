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
    topological_list = []
    topological_set = set()

    def dfs_helper(v: Variable):
        if v.unique_id in topological_set or v.is_constant():
            return

        for parent in v.parents:
            dfs_helper(parent)

        topological_set.add(v.unique_id)
        topological_list.append(v)

    dfs_helper(variable)
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
    unique_ids = [v.unique_id for v in variables]
    scalar_dict = {key: None for key in unique_ids}
    scalar_dict[variable.unique_id] = deriv

    # backpropagate
    for v in variables:
        if v.is_leaf():
            continue
        for key, item in v.chain_rule(scalar_dict[v.unique_id]):
            if scalar_dict[key.unique_id] is not None:
                scalar_dict[key.unique_id] += item
            else:
                scalar_dict[key.unique_id] = item

    # accumulate derivative for leaf
    for v in variables:
        if v.is_leaf():
            v.accumulate_derivative(scalar_dict[v.unique_id])


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
