from hypothesis.extra import numpy as np_st
from hypothesis import given
from hypothesis import strategies as st
import numpy as np
import operator as op
import functools as ft
from nibabel.cifti2.cifti2 import Cifti2Image, Cifti2Header
from ciftipy.main import CiftiImg
from tests import strategies as cp_st


@given(shape=np_st.array_shapes(max_dims=3), data=st.data())
def test_cifti_img_slice_same_as_nd_array_slice(shape, data: st.DataObject):
    array = np.linspace(0,ft.reduce(op.mul, shape)).reshape(shape)
    cifitpy_cifti_img = CiftiImg(
        Cifti2Image(dataobj=array.copy(), header=Cifti2Header())
    )
    ix = data.draw(cp_st.all_indicies(shape))

    assert np.ndarray(cifitpy_cifti_img[ix]) == array[ix]


@given(shape=np_st.array_shapes(max_dims=3), data=st.data())
def test_cifti_img_slice_same_shape_as_nd_array_slice(shape, data: st.DataObject):
    array = np.linspace(ft.reduce(op.mul, shape)).reshape(shape)
    cifitpy_cifti_img = CiftiImg(
        Cifti2Image(dataobj=array.copy(), header=Cifti2Header())
    )
    ix = data.draw(cp_st.all_indicies(shape))

    assert cifitpy_cifti_img[ix].shape == array[ix].shape