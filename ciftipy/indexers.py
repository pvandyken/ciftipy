from nibabel.cifti2 import cifti2_axes
import nibabel as nb
from typing import Any
from ciftipy.interfaces import nib as cp_nib


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
    # New BrainModelAxis
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
    # New BrainModelAxis
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


# def index_series_axis(
#     axis: cifti2_axes.SeriesAxis, index: CiftiIndex1d
# ):
#     # Parameters that need to be updated: name, meta
#     new_name = axis.name[index]
#     # The meta might be empty, which is reflected in an array with 1 empty dict
#     new_meta = axis.name[index]
#     # New BrainModelAxis
#     new_axis = nb.cifti2.cifti2_axes.ScalarAxis(new_name, new_meta)
#     return new_axis
