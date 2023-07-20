from __future__ import annotations
from typing import Any, Mapping, Sequence, SupportsIndex, TypeAlias, TypeVar
import numpy as np

ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)
NDArray: TypeAlias = np.ndarray[Any, np.dtype[ScalarType]]
CiftiMaskTypes: TypeAlias = NDArray[np.integer[Any]] | NDArray[np.bool_]
CiftiMaskIndex: TypeAlias = CiftiMaskTypes | tuple[CiftiMaskTypes, ...]
CiftiBasicIndexTypes: TypeAlias = (
    SupportsIndex | slice | ellipsis
)
CiftiBasicIndex: TypeAlias = CiftiBasicIndexTypes | tuple[CiftiBasicIndexTypes, ...]
CiftiIndex: TypeAlias = CiftiBasicIndex | CiftiMaskIndex

class CiftiIndexer:
    def __getitem__(self, __index: str) -> CiftiIndex: ...

    def __repr__(self) -> str: ...

class CiftiImg:
    @property
    def hemi(self) -> CiftiIndexer: ...
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
