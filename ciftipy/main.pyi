from __future__ import annotations
from typing import Any, Mapping, Sequence, SupportsIndex, TypeAlias, TypeVar
import numpy as np
from nibabel.cifti2 import cifti2, cifti2_axes

DType = TypeVar("DType", bound=np.dtype[Any])
ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)
NDArray: TypeAlias = np.ndarray[Any, np.dtype[ScalarType]]
CiftiMaskTypes: TypeAlias = NDArray[np.integer[Any]] | NDArray[np.bool_]
CiftiMaskIndex: TypeAlias = CiftiMaskTypes | tuple[CiftiMaskTypes, ...]
CiftiBasicIndexTypes: TypeAlias = SupportsIndex | slice | ellipsis
CiftiBasicIndex: TypeAlias = CiftiBasicIndexTypes | tuple[CiftiBasicIndexTypes, ...]
CiftiIndex1d: TypeAlias = CiftiBasicIndexTypes | CiftiMaskTypes
CiftiIndex: TypeAlias = CiftiBasicIndex | CiftiMaskIndex

class CiftiSearch:
    def __init__(self, bm_axis: cifti2_axes.BrainModelAxis) -> None: ...
    def __getitem__(self, __index: str) -> CiftiIndex: ...
    def __repr__(self) -> str: ...

class LabelMapping:
    def __init__(
        self, dataobj: np.ndarray, mapping_dict: dict, index_type: str
    ) -> None: ...
    def __getitem__(self, __index: str | int) -> CiftiIndex: ...

class CiftiImg:
    def __init__(self, cifti: cifti2.Cifti2Image) -> None: ...
    def __array__(self, dtype: DType) -> np.ndarray[Any, DType]: ...
    @property
    def search(self) -> CiftiSearch: ...
    @property
    def vertices(self) -> CiftiIndex: ...
    @property
    def voxels(self) -> CiftiIndex: ...
    @property
    def axis(self) -> Sequence[Axis]: ...
    @property
    def labels(self) -> LabelTable | None: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    def __getitem__(self, __index: CiftiIndex) -> CiftiImg: ...

class Axis: ...

class BrainModelAxis(Axis):
    @property
    def search(self) -> CiftiSearch: ...
    @property
    def vertices(self) -> CiftiIndex: ...
    @property
    def voxels(self) -> CiftiIndex: ...

class ParcelAxis(Axis):
    @property
    def search(self) -> CiftiSearch: ...
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
    def label(self) -> LabelMapping: ...
    @property
    def key(self) -> LabelMapping: ...

class SeriesAxis(list, Axis):
    name: str
    unit: str
    start: int
    step: int
    exponent: int
    size: int

class ScalarAxis:
    name: np.ndarray[Any, np.dtype[str]]
    meta: np.ndarray[Any, np.dtype[dict]]
