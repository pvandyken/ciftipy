from __future__ import annotations
import numpy as np
import nibabel as nb
from nibabel.cifti2 import cifti2
from typing import Any, Mapping, Sequence, SupportsIndex, TypeAlias, TypeVar
import more_itertools as itx
from abc import ABC

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


class CiftiSearch:
    def __getitem__(self, __index: str) -> CiftiIndex:
        ...

    def __repr__(self) -> str:
        ...


class LabelMapping:
    def __init__(self, dataobj: np.ndarray, mapping_dict: dict, index_type: str):
        self._dataobj = dataobj.astype(int)
        self._mapping = mapping_dict
        self._index_type = index_type

    def __getitem__(self, __index: str | int):
        if isinstance(__index, str) and self._index_type == "key":
            raise ValueError("Please provide an integer as key")
        elif isinstance(__index, str):
            return self._dataobj == self._mapping[__index]
        elif isinstance(__index, int) and self._index_type == "label":
            raise ValueError("Please provide an string if using labels as indexer.")
        elif isinstance(__index, int):
            return self._dataobj == __index
        else:
            raise Exception("Key can only be string or integer.")


class Axis(ABC):
    @property
    def size(self):
        return len(self)


class BrainModelAxis(Axis):
    def __init__(self, axis: nb.cifti2.cifti2_axes.BrainModelAxis):
        self._nb_axis = axis

    @property
    def search(self):
        return CiftiSearch(self._nb_axis)

    @property
    def vertices(self):
        # Returns an index mask
        # Based on original vertex object from Nibabel
        return self._nb_axis.vertex != -1

    @property
    def voxels(self):
        return self._nb_axis.voxel[:, 0] != -1

    def __len__(self):
        return len(self._nb_axis)


class ParcelAxis(Axis):
    def __init__(self, axis: nb.cifti2.cifti2_axes.ParcelsAxis):
        self._nb_axis = axis

    @property
    def search(self):
        return CiftiSearch2(self._nb_axis)

    @property
    def vertices(self) -> CiftiIndex:
        ...

    @property
    def voxels(self) -> CiftiIndex:
        ...

    def __len__(self):
        return len(self._nb_axis)


class LabelTableAxis(list, Axis):  # TypeAlias = "Sequence[LabelTable]"
    pass


class Label:
    def __init__(self, name, color):
        self.name = name
        self.color = color


class LabelTable(Mapping[str, Label]):
    def __init__(self, name, label, meta, dataobj):
        self.meta = meta
        self.name = name
        self.label = label
        self._dataobj = dataobj
        # Create a mapping from labels to keys
        mapp_dict = dict()
        for key in label:
            new_key = label[key][0]
            mapp_dict[new_key] = key
        self._mapping = mapp_dict

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta

    @meta.setter
    def meta(self, value):
        self._meta = value

    @property
    def label(self):
        return LabelMapping(self._dataobj, self._mapping, "label")

    @property
    def key(self):
        return LabelMapping(self._dataobj, self._mapping, "key")


class SeriesAxis(Axis):
    name: str
    unit: str
    start: int
    step: int
    exponent: int
    size: int


class ScalarAxis:
    def __init__(self, name, meta):
        self.name = name


class SeriesAxis(Axis):
    name: str
    unit: str
    start: int
    step: int
    exponent: int
    size: int


class ScalarAxis:
    @property
    def name(self) -> np.ndarray[Any, np.dtype[str]]:
        ...


class CiftiImg:
    def __init__(self, cifti: cifti2.Cifti2Image):
        self.nibabel_obj = cifti

    def __array__(self, dtype: DType = None):
        return self.nibabel_obj.get_fdata()

    @property
    def search(self):
        return

    @property
    def vertices(self) -> CiftiIndex:
        ...

    @property
    def voxels(self) -> CiftiIndex:
        ...

    @property
    def axis(self):
        # Get the axes from nibabel
        axes = [
            self.nibabel_obj.header.get_axis(i) for i in range(self.nibabel_obj.ndim)
        ]
        # Start building our axes
        new_axes = []
        for axis in axes:
            # Case 1: BrainModelAxis -> Column axis
            if isinstance(axis, nb.cifti2.cifti2_axes.BrainModelAxis):
                new_axes.append(BrainModelAxis(axis))
            # Case 2: ParcelsAxis -> Column axis
            elif isinstance(axis, nb.cifti2.cifti2_axes.ParcelsAxis):
                raise Exception("Parcel axis not supported for now")
            # Case 3: LabelAxis -> Row axis
            elif isinstance(axis, nb.cifti2.cifti2_axes.LabelAxis):
                tmp_axis = LabelTableAxis([])
                # In this case, we have to build the list of LabelTable objects
                for idx in range(len(axis.name)):
                    tmp_axis.append(
                        LabelTable(
                            axis.name[idx],
                            axis.label[idx],
                            axis.meta[idx],
                            self.nibabel_obj.get_fdata(),
                        )
                    )
                new_axes.append(LabelTableAxis(tmp_axis))
            # Case 4: LabelAxis -> Row axis
            elif isinstance(axis, nb.cifti2.cifti2_axes.ScalarAxis):
                new_axes.append(ScalarAxis(axis))
            # Case 5: SeriesAxis -> Row axis
            elif isinstance(axis, nb.cifti2.cifti2_axes.SeriesAxis):
                new_axes.append(SeriesAxis(axis))

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
                for labeltable_id in range(len(axis.name)):
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
