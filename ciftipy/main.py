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
        # Update the header

        # Construct a new object
        new_nb_obj =  nb.cifti2.Cifti2Image(new_data, self.nibabel_obj.header,
                                            self.nibabel_obj.nifti_header,
                                            self.nibabel_obj.extra,
                                            self.nibabel_obj.file_map)
        return CiftiImg(new_nb_obj)