from __future__ import annotations
import numpy as np
import nibabel as nb
from nibabel.cifti2 import cifti2, cifti2_axes
from typing import Any, Mapping, Sequence, SupportsIndex, TypeAlias, TypeVar
import more_itertools as itx

# from typing_extensions import Ellipsis
from ciftipy import indexers
from collections.abc import Iterable


DType = TypeVar("DType", bound=np.dtype[Any])
ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)
NDArray: TypeAlias = "np.ndarray[Any, np.dtype[ScalarType]]"
CiftiMaskTypes: TypeAlias = "NDArray[np.integer[Any]] | NDArray[np.bool_]"
CiftiMaskIndex: TypeAlias = "CiftiMaskTypes | tuple[CiftiMaskTypes, ...]"
CiftiBasicIndexTypes: TypeAlias = "SupportsIndex | slice | ellipsis"
CiftiBasicIndex: TypeAlias = "CiftiBasicIndexTypes | tuple[CiftiBasicIndexTypes, ...]"
CiftiIndex: TypeAlias = "CiftiBasicIndex | CiftiMaskIndex"


class CiftiIndexer:
    def __getitem__(self, __index: str) -> CiftiIndex:
        ...

    def __repr__(self) -> str:
        ...


class Axis:
    ...


class BrainModelAxis(Axis):
    @property
    def hemi(self) -> CiftiIndexer:
        ...

    @property
    def struc(self) -> CiftiIndexer:
        ...

    @property
    def vertices(self) -> CiftiIndex:
        ...

    @property
    def voxels(self) -> CiftiIndex:
        ...


class ParcelAxis(Axis):
    @property
    def hemi(self) -> CiftiIndexer:
        ...

    @property
    def struc(self) -> CiftiIndexer:
        ...

    @property
    def vertices(self) -> CiftiIndex:
        ...

    @property
    def voxels(self) -> CiftiIndex:
        ...


LabelTableAxis: TypeAlias = "Sequence[LabelTable]"


def wrap_axis(axis):
    if isinstance(axis, cifti2_axes):
        return BrainModelAxis(axis)
    return ...


class Label:
    name: str
    color: tuple[int, int, int, int]


class LabelTable(Mapping[str, Label]):
    name: str

    @property
    def meta(self) -> dict[str, Any]:
        ...

    @property
    def label(self) -> CiftiIndexer:
        ...

    @property
    def key(self) -> CiftiIndexer:
        ...


class CiftiImg:
    def __init__(self, cifti: cifti2.Cifti2Image):
        self.nibabel_obj = cifti

    def __array__(self, dtype: DType | None = None) -> np.ndarray[Any, DType]:
        if dtype is not None:
            return self.nibabel_obj.get_fdata().astype(dtype)
        return self.nibabel_obj.get_fdata()

    @property
    def hemi(self) -> CiftiIndexer:
        ...

    @property
    def struc(self) -> CiftiIndexer:
        ...

    @property
    def vertices(self) -> CiftiIndex:
        ...

    @property
    def voxels(self) -> CiftiIndex:
        ...

    @property
    def axis(self) -> Sequence[Axis]:
        return [
            wrap_axis(self.nibabel_obj.header.get_axis(i))
            for i in range(self.nibabel_obj.ndim)
        ]

    @property
    def labels(self) -> LabelTable | None:
        ...

    @property
    def shape(self):
        return self.nibabel_obj.shape

    def __getitem__(self, __index: CiftiIndex):
        # Get the data
        data = self.nibabel_obj.get_fdata()
        # Reformat __index
        __index = tuple(
            np.array(element).reshape(-1)
            if isinstance(element, Iterable) or isinstance(element, int)
            else element
            for element in itx.always_iterable(__index)
        )
        # Check case with booleans
        if isinstance(__index, tuple) and len(__index) > 1:
            # Check if any of the values is a boolean array
            bool_array = False
            non_array = False
            for element in __index:
                if isinstance(element, np.ndarray):
                    if element.dtype == bool:
                        bool_array = True
                else:
                    non_array = True
            # Raise error if there's a boolean array and an obj that is not an array
            if bool_array and non_array:
                raise Exception(
                    "All indexes must be arrays or integers if using a boolean mask for any dimension."
                )
            # If there's not a non-array, use ix_
            elif not non_array:
                _ixgrid = np.ix_(*__index)
                # Index the dataobj
                new_data = data[_ixgrid]
            else:
                # Get the shape of the dataobj
                array_shape = data.shape
                new_data = np.copy(data)
                # Iterate each axis at a time
                for idx, indexer in enumerate(__index):
                    slicer = tuple(
                        slice(0, size) if index != idx else indexer
                        for index, size in enumerate(array_shape)
                    )
                    new_data = new_data[slicer]
        # Case without tuples
        else:
            # Index the dataobj
            new_data = data[__index]
        # Update the header.
        if not isinstance(__index, tuple):  # or (isinstance(__index, np.ndarray)):
            __index = (__index,)
        for axis_idx, index_axis in enumerate(__index):
            # First get axes
            axis = self.nibabel_obj.header.get_axis(axis_idx)
            # Case 1: BrainModelAxis -> Column axis
            if isinstance(axis, nb.cifti2.cifti2_axes.BrainModelAxis):
                new_col_axis = indexers.index_brainmodel_axis(axis, index_axis)
            # Case 2: ParcelsAxis -> Column axis
            elif isinstance(axis, nb.cifti2.cifti2_axes.ParcelsAxis):
                new_col_axis = indexers.index_Parcels_axis(axis, index_axis)
            # Case 3: LabelAxis -> Row axis
            elif isinstance(axis, nb.cifti2.cifti2_axes.LabelAxis):
                new_row_axis = indexers.index_label_axis(axis, index_axis)
            # Case 4: LabelAxis -> Row axis
            elif isinstance(axis, nb.cifti2.cifti2_axes.ScalarAxis):
                new_row_axis = indexers.index_scalar_axis(axis, index_axis)
            # Case 5: SeriesAxis -> Row axis
            elif isinstance(axis, nb.cifti2.cifti2_axes.SeriesAxis):
                new_row_axis = indexers.index_series_axis(axis, index_axis)

        # Construct a new object
        new_nb_obj = nb.cifti2.Cifti2Image(
            new_data,
            self.nibabel_obj.header,
            self.nibabel_obj.nifti_header,
            self.nibabel_obj.extra,
            self.nibabel_obj.file_map,
            self.nibabel_obj.get_data_dtype(),
        )
        return CiftiImg(new_nb_obj)
