from typing import Any, Callable, TypeAlias, TypeVar
from typing_extensions import ParamSpec
import pytest
import functools as ft
import numpy as np

_T = TypeVar("_T")
_P = ParamSpec("_P")
_FuncT: TypeAlias = Callable[_P, _T]


def get_index_length(__ix: Any, length: int) -> int:

    """.. py:function:: get_index_length(__ix: Any, length: int) -> int
    
    Calculate the length of an index or slice.

    Parameters
    ----------
    __ix : Any
        The index or slice.
    length : int
        The length of the axis.

    Returns
    -------
    int
        The length of the index or slice.

    """
    if isinstance(__ix, tuple):
        if len(__ix):
            return get_index_length(__ix[0], length=length)
        return length
    if isinstance(__ix, slice):
        return len(range(*__ix.indices(length)))
    arr = np.array(__ix).reshape((-1,))
    if arr.dtype == np.bool_:
        return np.sum(arr)
    return arr.shape[0]

def debug(**overrides: Any):
    """Disable a hypothesis decorated test with specific parameter examples
    .. py:function:: debug
    
    
    Should be used as a decorator, placed *before* @hypothesis.given(). Adds the "debug"
    mark to the test, which can be detected by other constructs (e.g. to disable fakefs
    when hypothesis is not being used)

    Parameters
    ----------
    **overrides : Any
        

    Returns
    -------

    """

    def inner(func: _FuncT[_P, _T]) -> _FuncT[_P, _T]:

        """.. py:function:: inner_test(*args: _P.args, **kwargs: _P.kwargs)
        
            Decorator function that disables a hypothesis decorated test with specific parameter examples.

        Parameters
        ----------
        func : _FuncT[_P, _T]
            

        Returns
        -------
        _FuncT[_P,_T]
            The result of the inner test function.

        """
        if not hasattr(func, "hypothesis"):
            raise TypeError(f"{func} is not decorated with hypothesis.given")

        test = getattr(func, "hypothesis").inner_test

        @pytest.mark.disable_fakefs(True)
        @ft.wraps(func)
        def inner_test(*args: _P.args, **kwargs: _P.kwargs):

            """.. py:function:: inner_test(*args: _P.args, **kwargs: _P.kwargs)
            
                This function is a wrapper that calls the inner test function with the provided arguments and keyword arguments, along with any overrides. It is used as a decorator to disable a hypothesis decorated test with specific parameter examples.

            Parameters
            ----------
            *args : _P.args
                
            **kwargs : _P.kwargs
                

            Returns
            -------
            Any
                The result of the inner test function.

            """
            return test(*args, **{**kwargs, **overrides})

        return inner_test

    return inner


def mock_data(*draws: Any):
    """Utility function for mocking the hypothesis data strategy
    .. py:function:: mock_data
    
    
    Intended for combination with debug. Takes a list of values corresponding the draws
    taking from the data object. In the debug run, calls to data will return the given
    values in the order specified

    Parameters
    ----------
    *draws : Any
        

    Returns
    -------

    """

    # pylint: disable=missing-class-docstring, too-few-public-methods
    class MockData:

        """.. py:class:: MockData
        
            Utility class for mocking the hypothesis data strategy. It is used in combination with the `debug` function. This class takes a list of values corresponding to the draws taken from the data object. In the debug run, calls to the `data` object will return the given values in the order specified.

        Parameters
        ----------

        Returns
        -------

        """
        _draws = iter(draws)

        # pylint: disable=unused-argument
        def draw(self, strategy: Any, label: Any = None):

            """.. py:method:: draw(self, strategy: Any, label: Any = None)
            
                    This method is used to draw values from a strategy. It takes a strategy as input and returns the next value from the `_draws` attribute.

            Parameters
            ----------
            strategy : Any
                The strategy from which to draw values.
            label : Any
                An optional label for the drawn value.
                :return: The next value drawn from the strategy.
                :rtype: Any (Default value = None)

            Returns
            -------
            Any
                The next value drawn from the strategy.

            """
            return next(self._draws)

    return MockData()

