from hypothesis.extra import numpy as np_st
from hypothesis import strategies as st
import numpy as np

from nibabel.cifti2 import cifti2_axes, cifti2
from ciftipy.interfaces import nib as cp_nib
import functools as ft
import operator as op

from tests.structures import STRUCTURES
from ciftipy.main import CiftiImg


def all_indicies(shape: tuple[int, ...], *, allow_ellipsis: bool = True):
    return st.one_of(
        np_st.basic_indices(shape, allow_ellipsis=allow_ellipsis),
        np_st.integer_array_indices(shape, result_shape=np_st.array_shapes(max_dims=1)),
    )


def cifti_structures():
    return st.sampled_from(STRUCTURES)


@st.composite
def brain_model_axes(
    draw: st.DrawFn, *, names: st.SearchStrategy[str] = cifti_structures()
):
    types = np.array(draw(st.lists(st.booleans())), dtype=np.bool_)
    length = len(types)
    names = draw(np_st.arrays(np.object_, (length,), elements=names))
    voxels = draw(
        np_st.arrays(
            np.int16, (length, 3), elements=st.integers(min_value=0, max_value=20000)
        )
    )
    vertices = draw(
        np_st.arrays(
            np.int16, (length,), elements=st.integers(min_value=0, max_value=20000)
        )
    )
    voxels[types] = -1
    vertices[~types] = -1

    if np.all(voxels == -1):
        volume_shape = None
    elif voxels.shape:
        volume_shape = tuple(int(x) for x in np.max(voxels, axis=0))
    else:
        volume_shape = None
    return cp_nib.brain_model_axis(
        names=names,
        voxels=voxels,
        vertices=vertices,
        volume_shape=volume_shape,
        nvertices=dict(zip(*np.unique(names[vertices >= 0], return_counts=True))),
    )


def colors():
    return st.integers(min_value=0, max_value=100).map(lambda x: x / 100)


def rgbas():
    return st.tuples(colors(), colors(), colors(), colors())


def label_tables():
    return (
        st.dictionaries(st.text(max_size=10), rgbas(), min_size=1, max_size=20)
        .map(lambda d: dict(enumerate(d.items())))
    )



@st.composite
def realistic_brainmodel_axis(draw: st.DrawFn):
    vol_space = draw(np_st.array_shapes(min_dims=3, max_dims=3, max_side=3))
    mesh_spaces = draw(
        st.lists(
            np_st.array_shapes(min_dims=1, max_dims=1, max_side=3),
            min_size=1,
            max_size=2,
        )
    )
    mesh_names = draw(
        np_st.arrays(
            np.object_, (len(mesh_spaces),), elements=cifti_structures(), unique=True
        )
    )
    num_vols = draw(st.integers(min_value=1, max_value=2))
    vol_names = draw(
        np_st.arrays(
            np.object_,
            (num_vols,),
            elements=cifti_structures().filter(lambda s: s not in mesh_names),
            unique=True,
        )
    )
    mesh_masks = [
        draw(np_st.arrays(np.bool_, space).filter(np.any)) for space in mesh_spaces
    ]
    vol_masks = draw(
        np_st.arrays(
            np.bool_,
            (
                len(vol_names),
                *vol_space,
            ),
        ).filter(lambda arr: arr.reshape((len(vol_names), -1)).any(axis=1).all())
    )
    mesh_axes = [
        cifti2_axes.BrainModelAxis.from_mask(mask, name=name)
        for mask, name in zip(mesh_masks, mesh_names)
    ]
    vol_axes = [
        cifti2_axes.BrainModelAxis.from_mask(mask, name=name)
        for mask, name in zip(vol_masks, vol_names)
    ]
    return cp_nib.add_brain_model_axis(
        ft.reduce(cp_nib.add_brain_model_axis, mesh_axes, cp_nib.brain_model_axis()),
        ft.reduce(cp_nib.add_brain_model_axis, vol_axes, cp_nib.brain_model_axis()),
    )


@st.composite
def cifti_imgs(draw: st.DrawFn):
    axes = draw(st.lists(realistic_brainmodel_axis(), min_size=1, max_size=3))
    header = cifti2.Cifti2Header.from_axes(axes)
    shape = tuple(len(ax) for ax in axes)
    return CiftiImg(
        cifti2.Cifti2Image(
            dataobj=draw(np_st.arrays(np.int_, shape)),
            header=header,
        )
    )
