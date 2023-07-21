from hypothesis import strategies as st, given
from ciftipy.indexers import index_brainmodel_axis
from tests.helpers import get_index_length
import tests.strategies as cp_st
from nibabel.cifti2 import cifti2_axes
import numpy as np


class TestIndexBrainModelAxis:
    @given(data=st.data(), axis=cp_st.brain_model_axes().filter(len))
    def test_data_arrs_correct_length(
        self, data: st.DataObject, axis: cifti2_axes.BrainModelAxis
    ):
        ix = data.draw(cp_st.all_indicies((len(axis),), allow_ellipsis=False))
        length = get_index_length(ix, len(axis))
        res = index_brainmodel_axis(axis, ix)
        assert res.vertex.shape[0] == length
        assert res.name.shape[0] == length
        assert res.voxel.shape[0] == length

    @given(data=st.data(), axis=cp_st.brain_model_axes().filter(len))
    def test_voxels_plus_vertices_sum_to_correct_length(
        self, data: st.DataObject, axis: cifti2_axes.BrainModelAxis
    ):
        ix = data.draw(cp_st.all_indicies((len(axis),), allow_ellipsis=False))
        length = get_index_length(ix, len(axis))
        res = index_brainmodel_axis(axis, ix)
        assert (
            np.sum(res.vertex >= 0) + np.sum(np.all(res.voxel >= 0, axis=1)) == length
        )

    @given(data=st.data(), axis=cp_st.brain_model_axes().filter(len))
    def test_voxel_and_vertex_ix_dont_intersect(
        self, data: st.DataObject, axis: cifti2_axes.BrainModelAxis
    ):
        ix = data.draw(cp_st.all_indicies((len(axis),), allow_ellipsis=False))
        res = index_brainmodel_axis(axis, ix)
        assert np.logical_xor(res.vertex == -1, np.all(res.voxel == -1, axis=1)).all()
