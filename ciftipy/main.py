from __future__ import annotations
import numpy as np
import nibabel as nb
from nibabel.cifti2 import cifti2
from typing import Any, Mapping, Sequence, SupportsIndex, TypeAlias, TypeVar
# from typing_extensions import Ellipsis



DType = TypeVar("DType", bound=np.dtype[Any])
ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)
NDArray: TypeAlias = "np.ndarray[Any, np.dtype[ScalarType]]"
CiftiMaskTypes: TypeAlias = "NDArray[np.integer[Any]] | NDArray[np.bool_]"
CiftiMaskIndex: TypeAlias = "CiftiMaskTypes | tuple[CiftiMaskTypes, ...]"
CiftiBasicIndexTypes: TypeAlias = "SupportsIndex | slice | Ellipsis"
CiftiBasicIndex: TypeAlias = "CiftiBasicIndexTypes | tuple[CiftiBasicIndexTypes, ...]"
CiftiIndex: TypeAlias = "CiftiBasicIndex | CiftiMaskIndex"

class CiftiIndexer:
    def __getitem__(self, __index: str) -> CiftiIndex: ...
    def __repr__(self) -> str: ...
    

class Axis: ...

class BrainModelAxis(Axis):
    @property
    def hemi(self) -> CiftiIndexer: ...
    @property
    def struc(self) -> CiftiIndexer: ...
    @property
    def vertices(self) -> CiftiIndex: ...
    @property
    def voxels(self) -> CiftiIndex: ...

class ParcelAxis(Axis):
    @property
    def hemi(self) -> CiftiIndexer: ...
    @property
    def struc(self) -> CiftiIndexer: ...
    @property
    def vertices(self) -> CiftiIndex: ...
    @property
    def voxels(self) -> CiftiIndex: ...

LabelTableAxis: TypeAlias = "Sequence[LabelTable]"

class Label:
    name: str
    color: tuple[int, int, int, int]

class LabelTable(Mapping[str, Label]):
    name: str
    @property
    def meta(self) -> dict[str, Any]: ...
    @property
    def label(self) -> CiftiIndexer: ...
    @property
    def key(self) -> CiftiIndexer: ...

class CiftiImg:
    def __init__(self, cifti: cifti2.Cifti2Image):
        self.nibabel_obj = cifti

    def __array__(self, dtype: DType = None):
        return self.nibabel_obj.get_fdata()
    
    @property
    def hemi(self)-> CiftiIndexer: ...
    @property
    def struc(self) -> CiftiIndexer: ...
    @property
    def vertices(self) -> CiftiIndex: ...
    @property
    def voxels(self) -> CiftiIndex: ...
    @property
    def axis(self) -> Sequence[Axis]: ...
    @property
    def labels(self) -> LabelTable | None: ...
    @property
    def shape(self):
        return self.nibabel_obj.shape

    def __getitem__(self, __index: CiftiIndex):
        # Get the data
        data = self.nibabel_obj.get_fdata()
        # Index the dataobj
        new_data = data[__index]
        # Update the header. We only need to update attributes: name, voxel, vertices
        # volume_shape and nvertices don't have to change bc they are refering to the 
        # shape of the original volumetric file and the # vertices of the original gifti
        if not isinstance(__index, tuple): #or (isinstance(__index, np.ndarray)):
            __index = (__index,)
        for axis_idx, indexes in enumerate(__index):
            # First get axes
            axis = self.nibabel_obj.header.get_axis(axis_idx)
            # Case 1: BrainModelAxis
            if isinstance(axis, nb.cifti2.cifti2_axes.BrainModelAxis):
                # Subcase 1:
                if isinstance(indexes, slice):
                    # Updated params
                    new_name = axis.name[__index[1]] # Grabbing the second element of the tuple
                    new_vertex = axis.vertex[__index[1]]
                    new_voxel = axis.voxel[__index[1]]
                    # New BrainModelAxis 
                    new_bm_axis = nb.cifti2.cifti2_axes.BrainModelAxis(new_name, new_voxel, new_vertex,
                                                                    axis.affine, axis.volume_shape,
                                                                        axis.nvertices)
            elif isinstance(axis, <otherAxis>): # Work here, Mohamed
        # Construct a new object
        new_nb_obj =  nb.cifti2.Cifti2Image(new_data, self.nibabel_obj.header,
                                            self.nibabel_obj.nifti_header,
                                            self.nibabel_obj.extra,
                                            self.nibabel_obj.file_map)
        return CiftiImg(new_nb_obj)