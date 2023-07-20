from hypothesis.extra import numpy as np_st
from hypothesis import strategies as st
import numpy as np

from nibabel.cifti2 import cifti2_axes


def all_indicies(shape: tuple[int, ...]):
    return st.one_of(np_st.basic_indices(shape), np_st.integer_array_indices(shape))


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
    types = np.array(draw(st.lists(st.booleans(), min_size=1)), dtype=np.bool_)
    length = len(types)
    names = draw(np_st.arrays(np.object_, (length,), elements=cifti_structures()))
    voxels = draw(np_st.arrays(np.uint, (length, 3))).astype(np.int_)
    vertices = draw(np_st.arrays(np.uint, (length,))).astype(np.int_)
    voxels[types] = -1
    vertices[~types] = 1

    bma = object.__new__(cifti2_axes.BrainModelAxis)
    bma.name = names
    bma.voxel = voxels
    bma.vertex = vertices
    bma.affine = np.eye(4)

    bma.volume_shape = tuple(int(x) for x in np.max(voxels, axis=0))
    bma.nvertices = dict(zip(*np.unique(names[bma.vertex >= 0], return_counts=True)))
    return bma
