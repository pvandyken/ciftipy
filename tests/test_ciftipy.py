from hypothesis.extra import numpy as np_st
from hypothesis import strategies as st, given, settings, HealthCheck
import numpy as np
import operator as op
import functools as ft
from nibabel.cifti2.cifti2 import Cifti2Image, Cifti2Header
from ciftipy.main import CiftiImg
from tests import strategies as cp_st

@settings(suppress_health_check=[HealthCheck.too_slow])
@given(cifti=cp_st.cifti_imgs(), data=st.data())
def test_cifti_img_slice_same_as_nd_array_slice(cifti, data: st.DataObject):
    d = np.array(cifti).copy()
    ix = data.draw(cp_st.all_indicies((d.shape[0],), allow_ellipsis=False))

    assert np.all(np.array(cifti[ix]) == d[ix])


# @debug(shape=(1,),data=mock_data(0))
@settings(suppress_health_check=[HealthCheck.too_slow])
@given(cifti=cp_st.cifti_imgs(), data=st.data())
def test_slicing_cifti_img_slices_axes(cifti, data: st.DataObject):
    cifti_img = CiftiImg(cifti)
    d = np.array(cifti_img).copy()
    ix = data.draw(cp_st.all_indicies((d.shape[0],), allow_ellipsis=False))

    assert np.all(np.array(cifti_img[ix]) == d[ix])



@settings(suppress_health_check=[HealthCheck.too_slow])
@given(cifti=cp_st.cifti_imgs(), data=st.data())
def test_cifti_img_slice_same_shape_as_nd_array_slice(cifti, data: st.DataObject):
    cifti_img = CiftiImg(cifti)
    d = np.array(cifti_img).copy()
    ix = data.draw(cp_st.all_indicies((d.shape[0],), allow_ellipsis=False))

    assert cifti_img[ix].shape == d[ix].shape