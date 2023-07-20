from hypothesis.extra import numpy as np_st
from hypothesis import given
from hypothesis import strategies as st
import numpy as np
import operator as op
import functools as ft
from nibabel.cifti2.cifti2 import Cifti2Image, Cifti2Header
from ciftipy.main import CiftiImg


def all_indicies(shape: tuple[int, ...]):
    return st.one_of(np_st.basic_indices(shape), np_st.integer_array_indices(shape))
