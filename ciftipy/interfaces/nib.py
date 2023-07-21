from typing import Any
import numpy as np
from nibabel.cifti2 import cifti2_axes


def brain_model_axis(
    names: np.ndarray[Any, np.dtype[np.object_]] | None = None,
    voxels: np.ndarray[Any, np.dtype[np.int_]] | None = None,
    vertices: np.ndarray[Any, np.dtype[np.int_]] | None = None,
    affine: np.ndarray[Any, np.dtype[np.int_]] | None = None,
    volume_shape: tuple[int, int, int] | None = None,
    nvertices: dict[str, int] | None = None,
):
    bma = object.__new__(cifti2_axes.BrainModelAxis)
    bma.name = names
    bma.voxel = voxels
    bma.vertex = vertices
    if affine is None:
        bma.affine = np.eye(4)
    else:
        bma.affine = affine
    bma.volume_shape = volume_shape
    if nvertices is None:
        bma.nvertices = {}
    else:
        bma.nvertices = nvertices
    return bma

