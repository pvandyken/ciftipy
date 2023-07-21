from hypothesis.extra import numpy as np_st
from hypothesis import strategies as st
import numpy as np

from nibabel.cifti2 import cifti2_axes
from ciftipy.interfaces import nib as cp_nib


def all_indicies(shape: tuple[int, ...], *, allow_ellipsis: bool = True):
    return st.one_of(
        np_st.basic_indices(shape, allow_ellipsis=allow_ellipsis),
        np_st.integer_array_indices(shape),
    )


STRUCTURES = [
    "CIFTI_STRUCTURE_ACCUMBENS_LEFT",
    "CIFTI_STRUCTURE_ACCUMBENS_RIGHT",
    "CIFTI_STRUCTURE_ALL_WHITE_MATTER",
    "CIFTI_STRUCTURE_ALL_GREY_MATTER",
    "CIFTI_STRUCTURE_AMYGDALA_LEFT",
    "CIFTI_STRUCTURE_AMYGDALA_RIGHT",
    "CIFTI_STRUCTURE_BRAIN_STEM",
    "CIFTI_STRUCTURE_CAUDATE_LEFT",
    "CIFTI_STRUCTURE_CAUDATE_RIGHT",
    "CIFTI_STRUCTURE_CEREBELLAR_WHITE_MATTER_LEFT",
    "CIFTI_STRUCTURE_CEREBELLAR_WHITE_MATTER_RIGHT",
    "CIFTI_STRUCTURE_CEREBELLUM",
    "CIFTI_STRUCTURE_CEREBELLUM_LEFT",
    "CIFTI_STRUCTURE_CEREBELLUM_RIGHT",
    "CIFTI_STRUCTURE_CEREBRAL_WHITE_MATTER_LEFT",
    "CIFTI_STRUCTURE_CEREBRAL_WHITE_MATTER_RIGHT",
    "CIFTI_STRUCTURE_CORTEX",
    "CIFTI_STRUCTURE_CORTEX_LEFT",
    "CIFTI_STRUCTURE_CORTEX_RIGHT",
    "CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT",
    "CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT",
    "CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT",
    "CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT",
    "CIFTI_STRUCTURE_OTHER",
    "CIFTI_STRUCTURE_OTHER_GREY_MATTER",
    "CIFTI_STRUCTURE_OTHER_WHITE_MATTER",
    "CIFTI_STRUCTURE_PALLIDUM_LEFT",
    "CIFTI_STRUCTURE_PALLIDUM_RIGHT",
    "CIFTI_STRUCTURE_PUTAMEN_LEFT",
    "CIFTI_STRUCTURE_PUTAMEN_RIGHT",
    "CIFTI_STRUCTURE_THALAMUS_LEFT",
    "CIFTI_STRUCTURE_THALAMUS_RIGHT",
]


def cifti_structures():
    return st.sampled_from(STRUCTURES)


@st.composite
def brain_model_axes(draw: st.DrawFn):
    types = np.array(draw(st.lists(st.booleans())), dtype=np.bool_)
    length = len(types)
    names = draw(np_st.arrays(np.object_, (length,), elements=cifti_structures()))
    voxels = draw(
        np_st.arrays(np.int16, (length, 3), elements=st.integers(min_value=0, max_value=20000))
    )
    vertices = draw(
        np_st.arrays(np.int16, (length,), elements=st.integers(min_value=0, max_value=20000))
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
