from abc import ABC, abstractmethod

import numpy as np
import sympy as sym
from numpy.typing import NDArray


class Distribution(ABC):
    """
    **Generic** distribution.
    """

    def __init__(self, formula: sym.Expr):
        """
        Create a *Generic* distribution.

        :param formula: a symbolic formula
        """
        self._formula = formula

    def formula(self) -> sym.Expr:
        """
        :return: the symbolic formula of the distribution
        """
        return self._formula

    @property
    def to_latex(self) -> str:
        """
        :return: the latex representation of the distribution
        """
        return f"${sym.latex(self.formula)}$"

    def change_notation(self, theta_substitution: dict[str, str]) -> Distribution:
        """
        Change the notation of variables in the PDF function.

        :param theta_substitution: a dictionary mapping variable names into other names
        :return: a Distribution object having changed variable names
        """
        # Map vector variables into matrix symbols
        theta = {
            v: sym.MatrixSymbol(theta_substitution[str(v)], v.shape[0], v.shape[1])
            if isinstance(v, sym.MatrixSymbol)
            else theta_substitution[str(v)]
            for v in self._formula.free_symbols
        }

        return Distribution(self._formula.subs(theta))

    @abstractmethod
    def ml(self, x: NDArray[np.floating]) -> None:
        """
        Performs maximum likelihood estimation on the parameters using the given data.

        :param x: an (N, D) array of data values
        """
        raise NotImplementedError

    @abstractmethod
    def pdf(self, **kwargs) -> float | NDArray[np.floating]:
        """
        Compute the likelihood of the distribution on the given data, assuming that the
        data are independent and identically distributed.

        :param kwargs: a dictionary of variables to data values
        :return: the likelihood of the distribution
        """
        raise NotImplementedError

    @abstractmethod
    def draw(self, sample_size: int) -> NDArray[np.floating]:
        """
        Draw samples from the distribution.

        :param sample_size: the size of the sample
        :return: (N, D) array holding the samples
        """
        raise NotImplementedError(
            "Sampling from a generic distribution is not supported."
        )

    def __mul__(self, other: Distribution) -> Distribution:
        """
        Symbolic multiplication of distributions.

        :param other: another distribution
        :return: a Distribution object
        """
        return Distribution(self._formula * other.formula)
