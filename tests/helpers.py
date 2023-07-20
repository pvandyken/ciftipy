from typing import Any, Callable, TypeAlias, TypeVar
from typing_extensions import ParamSpec
import pytest
import functools as ft

_T = TypeVar("_T")
_P = ParamSpec("_P")
_FuncT: TypeAlias = Callable[_P, _T]

def debug(**overrides: Any):
    """Disable a hypothesis decorated test with specific parameter examples

    Should be used as a decorator, placed *before* @hypothesis.given(). Adds the "debug"
    mark to the test, which can be detected by other constructs (e.g. to disable fakefs
    when hypothesis is not being used)
    """

    def inner(func: _FuncT[_P, _T]) -> _FuncT[_P, _T]:
        if not hasattr(func, "hypothesis"):
            raise TypeError(f"{func} is not decorated with hypothesis.given")

        test = getattr(func, "hypothesis").inner_test

        @pytest.mark.disable_fakefs(True)
        @ft.wraps(func)
        def inner_test(*args: _P.args, **kwargs: _P.kwargs):
            return test(*args, **{**kwargs, **overrides})

        return inner_test

    return inner
