from __future__ import annotations
import textwrap
import numpy as np
import nibabel as nb
from nibabel.cifti2 import cifti2, cifti2_axes
from typing import Any, Mapping, Sequence, SupportsIndex, TypeAlias, TypeVar
import more_itertools as itx
from abc import ABC

# from typing_extensions import Ellipsis
import operator
from ciftipy import indexers
import ciftipy
from collections.abc import Iterable
import yaml
from yaml.loader import SafeLoader
import more_itertools as itx
from thefuzz import process
import os
from ciftipy import tokens


DType = TypeVar("DType", bound=np.dtype[Any])
ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)
NDArray: TypeAlias = "np.ndarray[Any, np.dtype[ScalarType]]"
CiftiMaskTypes: TypeAlias = "NDArray[np.integer[Any]] | NDArray[np.bool_]"
CiftiMaskIndex: TypeAlias = "CiftiMaskTypes | tuple[CiftiMaskTypes, ...]"
CiftiBasicIndexTypes: TypeAlias = "SupportsIndex | slice | ellipsis"
CiftiBasicIndex: TypeAlias = "CiftiBasicIndexTypes | tuple[CiftiBasicIndexTypes, ...]"
CiftiIndex: TypeAlias = "CiftiBasicIndex | CiftiMaskIndex"

with open(os.path.join(ciftipy.__path__[0], "search_tokens.yaml")) as f:
    search_tokens = yaml.load(f, Loader=SafeLoader)

for key in search_tokens.keys():
    search_tokens[key] = set(search_tokens[key])

all_search_tokens = np.array(
    list(
        search_tokens["token_left"]
        .union(search_tokens["token_right"])
        .union(search_tokens["token_other"])
    )
)


def load(path):
    return CiftiImg(nb.load(path))


class CiftiIndexHemi:
    def __init__(self, bm_axis: nb.cifti2.cifti2_axes.BrainModelAxis) -> None:
        self.size = bm_axis.size
        self.bm_structures = list(bm_axis.iter_structures())
        self.bm_structures_idxs = np.array(range(len(self.bm_structures)))
        self.bm_structures_names = np.fromiter(
            map(operator.itemgetter(0), self.bm_structures), dtype=np.object_
        )
        self.bm_structures_name_idx_dict = dict(
            zip(self.bm_structures_names, self.bm_structures_idxs)
        )
        self.bm_structures_slices = np.array(
            list(map(operator.itemgetter(1), self.bm_structures))
        )
        self.bm_axes = list(map(operator.itemgetter(2), self.bm_structures))

    def __getitem__(self, __index: str) -> CiftiIndex:
        # use fuzzer to get scores associated with either the 'left' or 'right' hemisphere
        scoresLR = process.extract(__index, ("left", "right"), limit=None)

        # checking fuzzer scores and getting set of standard cifti structures corresponding to hemispehere
        if scoresLR[0][1] > 50:
            if scoresLR[0][0] == "left":
                hemi_tokens = search_tokens["token_left"]
            elif scoresLR[0][0] == "right":
                hemi_tokens = search_tokens["token_right"]
        else:
            raise KeyError(f"Unrecognized query: {__index}")

        # getting indicies associated with hemisphere brainstructures
        kept_strucs = set(self.bm_structures_names).intersection(hemi_tokens)
        bm_indices = [self.bm_structures_name_idx_dict[struc] for struc in kept_strucs]

        # indexing list of brainstructures for indexed structures
        new_bm_structures = [self.bm_structures[ix] for ix in bm_indices]

        # get verticies from brains structures
        mask = np.zeros(self.size, dtype=np.bool_)  # Zeros
        for slice_ in [struc[1] for struc in new_bm_structures]:
            mask[slice_] = True

        return mask


class CiftiIndexStructure:
    def __init__(self, bm_axis: nb.cifti2.cifti2_axes.BrainModelAxis) -> None:
        self.size = bm_axis.size
        self.bm_structures = list(bm_axis.iter_structures())
        self.bm_structures_idxs = np.array(range(len(self.bm_structures)))
        self.bm_structures_names = np.array(
            list(map(operator.itemgetter(0), self.bm_structures))
        )
        self.bm_structures_name_idx_dict = dict(
            zip(self.bm_structures_names, self.bm_structures_idxs)
        )
        self.bm_structures_slices = np.array(
            list(map(operator.itemgetter(1), self.bm_structures))
        )
        self.bm_axes = list(map(operator.itemgetter(2), self.bm_structures))

    def __getitem__(self, __index: str) -> CiftiIndex:
        # use fuzzer to get scores associated with either the 'left' or 'right'
        # scoresLR = np.array(list(map(operator.itemgetter(1),process.extract(__index,('left','right'),limit=None))))

        strucs, scores = zip(
            *process.extract(__index, self.bm_structures_names, limit=None)
        )
        score_mask = np.array(scores) > 70
        kept_strucs = np.array(strucs)[score_mask]

        bm_indices = [self.bm_structures_name_idx_dict[struc] for struc in kept_strucs]

        # indexing list of brainstructures for indexed structures
        new_bm_structures = [self.bm_structures[ix] for ix in bm_indices]

        # get verticies from brains structures
        mask = np.zeros(self.size, dtype=np.bool_)  # Zeros
        for slice_ in [struc[1] for struc in new_bm_structures]:
            mask[slice_] = True

        return mask


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

    def __repr__(self):
        return (
            f"BrainModel: length={len(self)} "
            f"{{voxels={np.sum(self.voxels)}, vertices={np.sum(self.vertices)}}}"
        )

    @property
    def hemi(self):
        return CiftiIndexHemi(self._nb_axis)

    @property
    def struc(self):
        return CiftiIndexStructure(self._nb_axis)

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
        return None

    @property
    def vertices(self) -> CiftiIndex:
        ...

    @property
    def voxels(self) -> CiftiIndex:
        ...

    def __len__(self):
        return len(self._nb_axis)


class LabelTableAxis(list, Axis):  # TypeAlias = "Sequence[LabelTable]"
    def __repr__(self):
        tables = (f'"{table.name}"' for table in self)
        return f"LabelTables: [{', '.join(tables)}]"


def wrap_axis(axis):
    if isinstance(axis, cifti2_axes):
        return BrainModelAxis(axis)
    return ...


class Label:
    def __init__(self, name, color):
        self.name = name
        self.color = color


class LabelTable:
    def __init__(self, name, label, meta, dataobj):
        self.meta = meta
        self.name = name
        self.labels = label
        self._dataobj = dataobj
        # Create a mapping from labels to keys
        mapp_dict = dict()
        for key in label:
            new_key = label[key][0]
            mapp_dict[new_key] = key
        self._mapping = mapp_dict

    def __repr__(self):
        linewidth = np.get_printoptions()["linewidth"]
        canonical = super().__repr__()
        label_off = 8
        labels = textwrap.wrap(", ".join(self._mapping.keys()), linewidth - label_off)
        labels = (
            labels[0] + "\n" + textwrap.indent("\n".join(labels[1:]), " " * label_off)
        )
        return f"{canonical}\nname: {self.name}\nlabels: {labels}"

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


class ScalarAxis:
    def __init__(self, name, meta):
        self.name = name
        self.meta = meta


class SeriesAxis(Axis):
    def __init__(self, axis: nb.cifti2.cifti2_axes.SeriesAxis):
        self.unit = axis.unit
        self.start = axis.start
        self.step = axis.step
        self.length = axis.size
        self.exponent = axis.to_mapping(0).series_exponent


class CiftiImg:
    def __init__(self, cifti: cifti2.Cifti2Image):
        self.nibabel_obj = cifti
        # Get the axes from nibabel
        self._nb_axes = [
            self.nibabel_obj.header.get_axis(i) for i in range(self.nibabel_obj.ndim)
        ]

    def __array__(self, dtype: DType | None = None) -> np.ndarray[Any, DType]:
        if dtype is not None:
            return self.nibabel_obj.get_fdata().astype(dtype)
        return self.nibabel_obj.get_fdata()

    def __repr__(self):
        axes = "\n    ".join(map(repr, self.axis))
        result = super().__repr__()
        result = f"{result}\nDims: {len(self.shape)}\n    {axes}"
        if (
            not isinstance(self.nibabel_obj.dataobj, nb.arrayproxy.ArrayProxy)
            or not self.nibabel_obj.dataobj.is_proxy
        ):
            result = f"{result}\n\n{np.asanyarray(self.nibabel_obj.dataobj)}"
        else:
            result = f"{result}\n\n[... <data on file> ...]"
        return result

    @property
    def search(self):
        return

    @property
    def vertices(self) -> CiftiIndex:
        res = []
        for ax in self.axis:
            try:
                res.append(ax.vertices)
            except AttributeError:
                res.append(slice(None))
        return tuple(res)

    @property
    def voxels(self) -> CiftiIndex:
        res = []
        for ax in self.axis:
            try:
                res.append(ax.voxels)
            except AttributeError:
                res.append(slice(None))
        return tuple(res)

    @property
    def axis(self):
        # Start building our axes
        new_axes = []
        for axis in self._nb_axes:
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
                new_axes.append(ScalarAxis(axis.name, axis.meta))
            # Case 5: SeriesAxis -> Row axis
            elif isinstance(axis, nb.cifti2.cifti2_axes.SeriesAxis):
                new_axes.append(SeriesAxis(axis))
        return new_axes

    @property
    def labels(self) -> LabelTable | None:
        for axis in self._nb_axes:
            if isinstance(axis, nb.cifti2.cifti2_axes.LabelAxis):
                if len(axis.name) > 1:
                    return None
                else:  # return first label table
                    return LabelTable(
                        axis.name[0],
                        axis.label[0],
                        axis.meta[0],
                        self.nibabel_obj.get_fdata(),
                    )

    @property
    def shape(self):
        return self.nibabel_obj.shape

    def __getitem__(self, index: CiftiIndex, /):
        # Get the data
        data = self.nibabel_obj.get_fdata()
        # Reformat __index
        if not isinstance(index, tuple):
            index = (index,)
        orig_index = index
        index = tuple(
            element
            if isinstance(element, slice) or element is ...
            else np.array(element).reshape(-1)
            for element in index
        )
        has_array = any(isinstance(elem, np.ndarray) for elem in index)
        if has_array:
            index = tuple(
                np.r_[:shape][elem] if not isinstance(elem, np.ndarray) else elem
                for shape, elem in zip(data.shape, index)
            )
            index = np.ix_(*index)

        new_data = data[index]
        # Update the header.
        new_axes = []
        if not isinstance(index, tuple):  # or (isinstance(__index, np.ndarray)):
            index = (index,)
        for axis_idx, index_axis in enumerate(orig_index):
            # First get axes
            axis = self.nibabel_obj.header.get_axis(axis_idx)
            # Case 1: BrainModelAxis -> Column axis
            if isinstance(axis, nb.cifti2.cifti2_axes.BrainModelAxis):
                new_axes.append(indexers.index_brainmodel_axis(axis, index_axis))
            # Case 2: ParcelsAxis -> Column axis
            elif isinstance(axis, nb.cifti2.cifti2_axes.ParcelsAxis):
                new_axes.append(indexers.index_Parcels_axis(axis, index_axis))
            # Case 3: LabelAxis -> Row axis
            elif isinstance(axis, nb.cifti2.cifti2_axes.LabelAxis):
                new_axes.append(indexers.index_label_axis(axis, index_axis))
            # Case 4: LabelAxis -> Row axis
            elif isinstance(axis, nb.cifti2.cifti2_axes.ScalarAxis):
                new_axes.append(indexers.index_scalar_axis(axis, index_axis))
            # Case 5: SeriesAxis -> Row axis
            elif isinstance(axis, nb.cifti2.cifti2_axes.SeriesAxis):
                new_axes.append(indexers.index_series_axis(axis, index_axis))
        # New header
        new_header = nb.cifti2.cifti2.Cifti2Header.from_axes(new_axes)

        # Construct a new object
        new_nb_obj = nb.cifti2.Cifti2Image(
            new_data,
            new_header,
            self.nibabel_obj.nifti_header,
            self.nibabel_obj.extra,
            self.nibabel_obj.file_map,
            self.nibabel_obj.get_data_dtype(),
        )
        return CiftiImg(new_nb_obj)
