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

    """.. py:function:: all_indicies(shape: tuple[int, ...], *, allow_ellipsis: bool = True)
    
    Generates strategies for generating indices for a given shape.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the array for which indices are generated.
    * :
        
    allow_ellipsis : bool
        Whether to allow the use of ellipsis in the indices. Defaults to True.

    Returns
    -------
    hypothesis.strategies.SearchStrategy
        A strategy for generating indices for the given shape.

    """
    return st.one_of(
        np_st.basic_indices(shape, allow_ellipsis=allow_ellipsis),
        np_st.integer_array_indices(shape, result_shape=np_st.array_shapes(max_dims=1)),
    )


def cifti_structures():

    """.. py:function:: cifti_structures()
    
    Returns a strategy that samples from a list of CIFTI structures.

    Parameters
    ----------

    Returns
    -------
    hypothesis.strategies.SearchStrategy[str]
        A strategy that samples from a list of CIFTI structures.

    """
    return st.sampled_from(STRUCTURES)


@st.composite
def brain_model_axes(
    draw: st.DrawFn, *, names: st.SearchStrategy[str] = cifti_structures()
):

    """.. py:function:: brain_model_axes(draw: st.DrawFn, *, names: st.SearchStrategy[str] = cifti_structures())
    
    This function generates a brain model axis using the provided strategies.

    Parameters
    ----------
    draw : st.DrawFn
        The draw function from the hypothesis library.
    * :
        
    names : st.SearchStrategy[str]
        The search strategy for generating names for the brain model axis. Defaults to cifti_structures().

    Returns
    -------
    cifti2_axes.BrainModelAxis
        The generated brain model axis.

    """
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

    """.. py:function:: colors()
    
    Generates random integers between 0 and 100 and maps them to values between 0 and 1.

    Parameters
    ----------

    Returns
    -------
    float
        A random integer between 0 and 100 mapped to a value between 0 and 1.

    """
    return st.integers(min_value=0, max_value=100).map(lambda x: x / 100)


def rgbas():

    """.. py:function:: rgbas()
    
    Generates a strategy for generating tuples of RGBA values.

    Parameters
    ----------

    Returns
    -------
    hypothesis.strategies.SearchStrategy[tuple[float, float, float, float]]
        A strategy for generating tuples of RGBA values.

    """
    return st.tuples(colors(), colors(), colors(), colors())


def label_tables():

    """.. py:function:: label_tables()
    
    Generates a strategy for creating dictionaries with string keys and RGBA values. The dictionaries have a minimum size of 1 and a maximum size of 20. Each key is a string with a maximum size of 10 characters, and each value is an RGBA tuple.

    Parameters
    ----------

    Returns
    -------
    hypothesis.strategies.SearchStrategy[dict[str, Tuple[float, float, float, float]]]
        A strategy for generating dictionaries with string keys and RGBA values.

    """
    return (
        st.dictionaries(st.text(max_size=10), rgbas(), min_size=1, max_size=20)
        .map(lambda d: dict(enumerate(d.items())))
    )



@st.composite
def realistic_brainmodel_axis(draw: st.DrawFn):

    """.. py:function:: realistic_brainmodel_axis(draw: st.DrawFn)
    
    This function generates a realistic brain model axis using the provided draw function.

    Parameters
    ----------
    draw : st.DrawFn
        The draw function used to generate random values.

    Returns
    -------
    cifti2_axes.BrainModelAxis
        A brain model axis representing the realistic brain model.

    """
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

    """.. py:function:: cifti_imgs(draw: st.DrawFn)
    
    Generate a CiftiImg object with realistic brain model axes.

    Parameters
    ----------
    draw : st.DrawFn
        A function used to draw values from the given strategies.

    Returns
    -------
    CiftiImg
        A CiftiImg object with realistic brain model axes.

    """
    axes = draw(st.lists(realistic_brainmodel_axis(), min_size=1, max_size=3))
    header = cifti2.Cifti2Header.from_axes(axes)
    shape = tuple(len(ax) for ax in axes)
    return CiftiImg(
        cifti2.Cifti2Image(
            dataobj=draw(np_st.arrays(np.int_, shape)),
            header=header,
        )
    )
