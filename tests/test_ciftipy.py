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
    array_length = ft.reduce(op.mul, shape)
    array = np.linspace(array_length).reshape(shape)

    cifti_img = Cifti2Image(dataobj=array.copy(), header=Cifti2Header())

    cifitpy_cifti_img = CiftiImg(cifti_img)

    ix = data.draw(cp_st.all_indicies(shape))

    assert np.ndarray(cifitpy_cifti_img[ix]) == array[ix]
