from typing import Any
import numpy as np
from nibabel.cifti2 import cifti2_axes


def add_brain_model_axis(
    first: cifti2_axes.BrainModelAxis, other: cifti2_axes.BrainModelAxis
):
    """.. py:function:: add_brain_model_axis
    
        Concatenates two BrainModels

    Parameters
    ----------
    first : cifti2_axes.BrainModelAxis
        
    other : cifti2_axes.BrainModelAxis
        brain model to be appended to the current one

    Returns
    -------
    BrainModelAxis
        

    
    """
    if first.affine is None:
        affine, shape = other.affine, other.volume_shape
    else:
        affine, shape = first.affine, first.volume_shape
        if other.affine is not None and (
            not np.allclose(other.affine, affine) or other.volume_shape != shape
        ):
            raise ValueError(
                "Trying to concatenate two BrainModels defined in a different brain volume"
            )

    nvertices = dict(first.nvertices)
    for name, value in other.nvertices.items():
        if name in nvertices.keys() and nvertices[name] != value:
            raise ValueError(
                "Trying to concatenate two BrainModels with "
                f"inconsistent number of vertices for {name}"
            )
        nvertices[name] = value
    return brain_model_axis(
        names=np.append(first.name, other.name),
        voxels=np.concatenate((first.voxel, other.voxel), 0),
        vertices=np.append(first.vertex, other.vertex),
        affine=affine,
        volume_shape=shape,
        nvertices=nvertices,
    )


def brain_model_axis(
    names: np.ndarray[Any, np.dtype[np.object_]] = np.ndarray((0,)),
    voxels: np.ndarray[Any, np.dtype[np.int_]] = np.ndarray((0, 3)),
    vertices: np.ndarray[Any, np.dtype[np.int_]] = np.ndarray((0,)),
    affine: np.ndarray[Any, np.dtype[np.int_]] | None = None,
    volume_shape: tuple[int, int, int] | None = None,
    nvertices: dict[str, int] | None = None,
):
    """.. py:function:: brain_model_axis(names: np.ndarray[Any, np.dtype[np.object_]] = np.ndarray((0,)), voxels: np.ndarray[Any, np.dtype[np.int_]] = np.ndarray((0, 3)), vertices: np.ndarray[Any, np.dtype[np.int_]] = np.ndarray((0,)), affine: np.ndarray[Any, np.dtype[np.int_]] | None = None, volume_shape: tuple[int, int, int] | None = None, nvertices: dict[str, int] | None = None) -> cifti2_axes.BrainModelAxis
    
        Create a new instance of the BrainModelAxis class with the specified parameters.

    Parameters
    ----------
    names : np.ndarray[Any, np.dtype[np.object_]]
        An array of names for the brain model axis. Default is an empty array.
    voxels : np.ndarray[Any, np.dtype[np.int_]]
        An array of voxel coordinates for the brain model axis. Default is an empty array of shape (0, 3).
    vertices : np.ndarray[Any, np.dtype[np.int_]]
        An array of vertex indices for the brain model axis. Default is an empty array.
    affine : np.ndarray[Any, np.dtype[np.int_]] | None
        An affine transformation matrix for the brain model axis. Default is None.
    volume_shape : tuple[int, int, int] | None
        The shape of the original volumetric file. Default is None.
    nvertices : dict[str, int] | None
        A dictionary mapping names to the number of vertices for each name. Default is None.
        :return: A new instance of the BrainModelAxis class with the specified parameters.
        :rtype: cifti2_axes.BrainModelAxis

    Returns
    -------
    cifti2_axes.BrainModelAxis
        A new instance of the BrainModelAxis class with the specified parameters.

    """
    bma = object.__new__(cifti2_axes.BrainModelAxis)
    bma.name = names
    bma.voxel = voxels
    bma.vertex = vertices
    bma.affine = affine
    bma.volume_shape = volume_shape
    if nvertices is None:
        bma.nvertices = {}
    else:
        bma.nvertices = nvertices
    return bma
