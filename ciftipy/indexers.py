from nibabel.cifti2 import cifti2_axes
import nibabel as nb
from typing import Any
from ciftipy.interfaces import nib as cp_nib
from collections.abc import Iterable
import numpy as np


def index_brainmodel_axis(axis: cifti2_axes.BrainModelAxis, index: Any):  # CiftiIndex1d
    # Updated params: name, vertex, voxel
    # We only need to update attributes: name, voxel, vertices
    # volume_shape and nvertices don't have to change bc they are refering to the
    # shape of the original volumetric file and the # vertices of the original gifti
    # New BrainModelAxis
    return cp_nib.brain_model_axis(
        names=axis.name[index].reshape((-1,)),
        vertices=axis.vertex[index].reshape((-1,)),
        voxels=axis.voxel[index].reshape((-1, 3)),
        affine=axis.affine,
        volume_shape=axis.volume_shape,
        nvertices=axis.nvertices,
    )


def index_parcel_axis(axis: cifti2_axes.ParcelsAxis, index: Any):  # CiftiIndex1d
    # Parameters that need to be updated: name, voxel, vertices
    new_name = axis.name[index]
    new_vertices = axis.vertices[index]
    new_voxels = axis.voxels[index]
    # New ParcelsAxis
    new_axis = nb.cifti2.cifti2_axes.ParcelsAxis(
        new_name,
        new_voxels,
        new_vertices,
        axis.affine,
        axis.volume_shape,
        axis.nvertices,
    )
    return new_axis


def index_scalar_axis(axis: cifti2_axes.ScalarAxis, index: Any):  # CiftiIndex1d
    # Parameters that need to be updated: name, meta
    new_name = axis.name[index]
    # The meta might be empty, which is reflected in an array with 1 empty dict
    new_meta = axis.name[index]
    # New ScalarAxis
    new_axis = nb.cifti2.cifti2_axes.ScalarAxis(new_name, new_meta)
    return new_axis


def index_label_axis(axis: cifti2_axes.LabelAxis, index: Any):
    index_mapping = axis.from_index_mapping
    new_label = axis.label[index]
    new_meta = axis.meta[index]
    new_name = axis.name[index]

    new_axis = nb.cifti2.cifti2_axes.LabelAxis(
        name=new_name, label=new_label, meta=new_meta
    )

    return new_axis


def index_series_axis(axis: cifti2_axes.SeriesAxis, index: Any):  # CiftiIndex1d
    # Parameters that need to be updated: start, size
    # Here it's necessary to have subcases for indexes: arrays or slices
    # First case: iterables
    if isinstance(index, np.ndarray):
        # Check if it's a mask by checking dtype
        if index.dtype == bool:
            # Build indexes based on original structure
            indexes_series = np.arange(axis.size)
            # Convert to indexes
            indexes_chosen = indexes_series[index]
            # Get the first element to get the start time
            new_start = axis.start + axis.step * indexes_chosen[0]
            # Get the size as the total number of indexes chosen
            new_size = indexes_chosen.size
        # Array index
        else:
            # Get the first element to get the start time
            new_start = axis.start + axis.step * index[0]
            # Get the size as the total number of indexes chosen
            new_size = index.size
    # Second case: slices
    else:
        # Get the first element to get the start time
        new_start = axis.start + axis.step * index.start
        # Get the size as the total number of indexes chosen
        new_size = len(range(*index.indices(axis.size)))
    # New SeriesAxis
    new_axis = nb.cifti2.cifti2_axes.SeriesAxis(
        new_start, axis.step, new_size, axis.unit
    )
    return new_axis
