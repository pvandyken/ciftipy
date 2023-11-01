from __future__ import annotations
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

    """.. py:class:: CiftiIndexHemi

    This class represents the hemisphere index of a Cifti image. It provides methods and properties to access and manipulate the hemisphere index.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, bm_axis: nb.cifti2.cifti2_axes.BrainModelAxis) -> None:
        """
        .. py:method:: CiftiIndexStructure.__init__(bm_axis: nb.cifti2.cifti2_axes.BrainModelAxis) -> None

            Initializes a CiftiIndexStructure object.

            :param bm_axis: The BrainModelAxis object.
            :type bm_axis: nb.cifti2.cifti2_axes.BrainModelAxis
            :return: None
            :rtype: None
        """
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
        """
        .. py:method:: CiftiIndexHemi.__getitem__(__index: str) -> np.ndarray

            This method is used to retrieve a mask of vertices associated with a specific hemisphere in a CiftiIndexHemi object.

            :param __index: The index used to determine the hemisphere. It can be either "left" or "right".
            :type __index: str
            :return: A boolean mask indicating the vertices associated with the specified hemisphere.
            :rtype: np.ndarray
        """
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
            print(slice_)
            mask[slice_] = True

        return mask


class CiftiIndexStructure:

    """.. py:class:: CiftiIndexStructure

    This class represents the index structure for a CIFTI file. It stores information about the brain model axis, including its size, structures, and axes.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, bm_axis: nb.cifti2.cifti2_axes.BrainModelAxis) -> None:
        """
        .. py:method:: CiftiIndexStructure.__init__(bm_axis: nb.cifti2.cifti2_axes.BrainModelAxis) -> None

            Initializes a CiftiIndexStructure object.

            :param bm_axis: The BrainModelAxis object.
            :type bm_axis: nb.cifti2.cifti2_axes.BrainModelAxis
            :return: None
            :rtype: None
        """
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
        """
        .. py:method:: __getitem__(__index: str) -> np.ndarray

            This method is used to retrieve a subset of the data based on the given index.

            :param __index: The index used to select the subset of data.
            :type __index: str
            :return: The subset of data.
            :rtype: np.ndarray
        """
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
            print(slice_)
            mask[slice_] = True

        return mask


class LabelMapping:

    """.. py:class:: LabelMapping

    This class represents a mapping between labels and keys in a Cifti image. It is used to map labels to their corresponding keys in the data object.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, dataobj: np.ndarray, mapping_dict: dict, index_type: str):
        """
        .. py:method:: __init__(dataobj: np.ndarray, mapping_dict: dict, index_type: str)

            Initializes a new instance of the LabelMapping class.

            :param dataobj: The data object to be stored in the LabelMapping instance.
            :type dataobj: np.ndarray
            :param mapping_dict: The mapping dictionary to be stored in the LabelMapping instance.
            :type mapping_dict: dict
            :param index_type: The type of index to be stored in the LabelMapping instance.
            :type index_type: str
            :return: None
            :rtype: None
        """
        self._dataobj = dataobj.astype(int)
        self._mapping = mapping_dict
        self._index_type = index_type

    def __getitem__(self, __index: str | int):
        """
        .. py:method:: __getitem__(self, __index: str | int)

            This method is used to retrieve an item from the data object based on the given index. The index can be either a string or an integer.

            :param __index: The index used to retrieve the item from the data object.
            :type __index: str or int
            :return: The item from the data object that matches the given index.
            :rtype: Any

            :raises ValueError: If the index is a string and the index type is "key", a ValueError is raised.
            :raises ValueError: If the index is an integer and the index type is "label", a ValueError is raised.
            :raises Exception: If the index is not a string or an integer, an Exception is raised.
        """
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

    """.. py:class:: Axis

    This class represents an abstract base class for different types of axes in a Cifti image. It provides common properties and methods for all axis types.

    Parameters
    ----------

    Returns
    -------

    """

    @property
    def size(self):
        """.. py:method:: size()

        Returns the size of the object.

        Parameters
        ----------

        Returns
        -------
        int
            The size of the object.

        """
        return len(self)


class BrainModelAxis(Axis):

    """.. py:class:: BrainModelAxis

    Represents a brain model axis in a CIFTI file.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, axis: nb.cifti2.cifti2_axes.BrainModelAxis):
        """
        .. py:method:: __init__(axis)

            Initializes a new instance of the Axis class.

            :param axis: The axis object to be assigned to the Axis instance.
            :type axis: nb.cifti2.cifti2_axes.Axis
            :return: None
            :rtype: None
        """
        self._nb_axis = axis

    def __repr__(self):
        """
        .. py:method:: __repr__()

            Returns a string representation of the `BrainModelAxis` object.

            :return: A string representation of the `BrainModelAxis` object.
            :rtype: str
        """
        return (
            f"BrainModel: length={len(self)} "
            f"{{voxels={np.sum(self.voxels)}, vertices={np.sum(self.vertices)}}}"
        )

    @property
    def hemi(self):
        """.. py:method:: hemi

        Returns a CiftiIndexHemi object.

        Parameters
        ----------

        Returns
        -------
        CiftiIndexHemi
            A CiftiIndexHemi object.

        """
        return CiftiIndexHemi(self._nb_axis)

    @property
    def struc(self):
        """.. py:method:: CiftiIndexStructure.__init__(self, bm_axis: nb.cifti2.cifti2_axes.BrainModelAxis) -> None

        Initializes a CiftiIndexStructure object.

        Parameters
        ----------

        Returns
        -------
        None
            None

        """
        return CiftiIndexStructure(self._nb_axis)

    @property
    def vertices(self):
        """.. py:method:: vertices

        Returns an index mask based on the original vertex object from Nibabel.

        Parameters
        ----------

        Returns
        -------
        numpy.ndarray
            An index mask.

        """
        # Returns an index mask
        # Based on original vertex object from Nibabel
        return self._nb_axis.vertex != -1

    @property
    def voxels(self):
        """.. py:method:: voxels()

        This method returns a boolean array indicating which voxels are valid. A voxel is considered valid if its value in the first column of the `_nb_axis.voxel` array is not equal to -1.

        Parameters
        ----------

        Returns
        -------
        numpy.ndarray
            A boolean array indicating which voxels are valid.

        """
        return self._nb_axis.voxel[:, 0] != -1

    def __len__(self):
        """
        .. py:method:: __len__()

            Returns the length of the `_nb_axis` attribute.

            :return: The length of the `_nb_axis` attribute.
            :rtype: int
        """
        return len(self._nb_axis)


class ParcelAxis(Axis):

    """.. py:class:: ParcelAxis

    Represents an axis for parcels in a Cifti image.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, axis: nb.cifti2.cifti2_axes.ParcelsAxis):
        """
        .. py:method:: __init__(axis)

            Initializes a new instance of the Axis class.

            :param axis: The axis object to be assigned to the Axis instance.
            :type axis: nb.cifti2.cifti2_axes.Axis
            :return: None
            :rtype: None
        """
        self._nb_axis = axis

    @property
    def search(self):
        """.. py:method:: search()

        This method is used to perform a search. However, the implementation of the search functionality is not provided in the code snippet. Therefore, this method currently returns None.

        Parameters
        ----------

        Returns
        -------
        None
            None

        """
        return None

    @property
    def vertices(self) -> CiftiIndex:
        """.. py:method:: vertices(self) -> CiftiIndex

        Returns the vertices of the brain model axis.

        Parameters
        ----------

        Returns
        -------
        CiftiIndex
            The vertices of the brain model axis.

        """
        ...

    @property
    def voxels(self) -> CiftiIndex:
        """.. py:method:: voxels

        This method returns the voxels of the brain model axis. Voxels are the discrete 3D units that make up the volume of the brain. The voxels represent the spatial locations within the brain where data is measured or analyzed.

        Parameters
        ----------

        Returns
        -------
        CiftiIndex
            A CiftiIndex object representing the voxels of the brain model axis.

        """
        ...

    def __len__(self):
        """
        .. py:method:: __len__()

            Returns the length of the `_nb_axis` attribute.

            :return: The length of the `_nb_axis` attribute.
            :rtype: int
        """
        return len(self._nb_axis)


class LabelTableAxis(list, Axis):  # TypeAlias = "Sequence[LabelTable]"
    pass


def wrap_axis(axis):
    """.. py:class:: LabelTableAxis(list, Axis)

    A class that represents an axis for a label table in a Cifti image.

    Parameters
    ----------

    Returns
    -------

    """

    """
.. py:function:: wrap_axis(axis)

    Wraps the given axis object in a BrainModelAxis if it is an instance of cifti2_axes.

    :param axis: The axis object to be wrapped.
    :type axis: cifti2_axes
    :return: The wrapped axis object.
    :rtype: BrainModelAxis

"""
    if isinstance(axis, cifti2_axes):
        return BrainModelAxis(axis)
    return ...


class Label:

    """.. py:class:: Label

    Represents a label with a name and color.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, name, color):
        """
        .. py:method:: __init__(name, color)

            Initializes a Label object with a name and color.

            :param name: The name of the label.
            :type name: str
            :param color: The color of the label.
            :type color: str
            :return: None
            :rtype: None
        """
        self.name = name
        self.color = color


class LabelTable:

    """.. py:class:: LabelTable

    Represents a table of labels associated with a Cifti image.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, name, label, meta, dataobj):
        """
        .. py:method:: __init__(self, name, label, meta, dataobj)

            Initializes a new instance of the LabelTable class.

            :param name: The name of the label table.
            :type name: str
            :param label: The labels in the label table.
            :type label: dict[str, List[int]]
            :param meta: The metadata associated with the label table.
            :type meta: dict[str, Any]
            :param dataobj: The data object associated with the label table.
            :type dataobj: np.ndarray
            :return: None
            :rtype: None

            This method initializes a new instance of the LabelTable class. It sets the name, labels, metadata, and data object of the label table. It also creates a mapping from labels to keys using the label dictionary.
        """
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
        canonical = super().__repr__()
        return (
            f"{canonical}\nLabel Table\n-----------\n    name: {self.name}\n    labels: "
            + ", ".join(self._mapping.keys())
        )

    @property
    def meta(self) -> dict[str, Any]:
        """.. py:method:: meta

        Returns the metadata associated with the LabelTable object.

        Parameters
        ----------

        Returns
        -------
        dict[str,Any]
            The metadata associated with the LabelTable object.

        """
        return self._meta

    @meta.setter
    def meta(self, value):
        """.. py:method:: meta

        Sets the metadata of the LabelTable.

        Parameters
        ----------
        value : dict[str, Any]
            The metadata to set.

        Returns
        -------
        type
            None.

        """
        self._meta = value

    @property
    def label(self):
        """.. py:method:: LabelMapping.__init__(self, dataobj: np.ndarray, mapping_dict: dict, index_type: str)

        The `LabelMapping` class constructor initializes a `LabelMapping` object.

        Parameters
        ----------

        Returns
        -------
        LabelMapping
            A `LabelMapping` object.

        """
        return LabelMapping(self._dataobj, self._mapping, "label")

    @property
    def key(self):
        """.. py:method:: key()

        This method returns a LabelMapping object with the data object, mapping dictionary, and index type set to "key".

        Parameters
        ----------

        Returns
        -------
        LabelMapping
            A LabelMapping object with the data object, mapping dictionary, and index type set to "key".

        """
        return LabelMapping(self._dataobj, self._mapping, "key")


class ScalarAxis:

    """.. py:class:: ScalarAxis

    Represents a scalar axis in a Cifti image.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, name, meta):
        """
        .. py:method:: __init__(name, meta)

            Initializes a new instance of the LabelTable class.

            :param name: The name of the label table.
            :type name: str
            :param meta: The metadata associated with the label table.
            :type meta: dict[str, Any]
            :return: None
            :rtype: None
        """
        self.name = name
        self.meta = meta


class SeriesAxis(Axis):

    """.. py:class:: SeriesAxis(Axis)

    Represents a series axis in a Cifti image.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, axis: nb.cifti2.cifti2_axes.SeriesAxis):
        """
        .. py:method:: __init__(axis: nb.cifti2.cifti2_axes.SeriesAxis)

            Initializes a new instance of the SeriesAxis class.

            :param axis: The original cifti2_axes.SeriesAxis object.
            :type axis: nb.cifti2.cifti2_axes.SeriesAxis
            :return: None
            :rtype: None
        """
        self.unit = axis.unit
        self.start = axis.start
        self.step = axis.step
        self.length = axis.size
        self.exponent = axis.to_mapping(0).series_exponent


class CiftiImg:

    """.. py:class:: CiftiImg

    This class represents a CIFTI image object. It encapsulates a Cifti2Image object from the nibabel library and provides convenient methods for accessing and manipulating the data and metadata of the CIFTI image.

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, cifti: cifti2.Cifti2Image):
        """
        .. py:method:: __init__(cifti: cifti2.Cifti2Image)

            Initializes a new CiftiImg object.

            :param cifti: The Cifti2Image object.
            :type cifti: cifti2.Cifti2Image
            :return: None
            :rtype: None
        """
        self.nibabel_obj = cifti
        # Get the axes from nibabel
        self._nb_axes = [
            self.nibabel_obj.header.get_axis(i) for i in range(self.nibabel_obj.ndim)
        ]

    def __array__(self, dtype: DType | None = None) -> np.ndarray[Any, DType]:
        """
        .. py:method:: __array__(self, dtype: DType | None = None) -> np.ndarray[Any, DType]

            This method returns the data from the `nibabel_obj` attribute as a NumPy array. If the `dtype` parameter is not None, the data is cast to the specified data type before returning.

            :param dtype: The data type to cast the data to. If None, the data is returned as is.
            :type dtype: DType | None
            :return: The data from the `nibabel_obj` attribute as a NumPy array.
            :rtype: np.ndarray[Any, DType]
        """
        if dtype is not None:
            return self.nibabel_obj.get_fdata().astype(dtype)
        return self.nibabel_obj.get_fdata()

    def __repr__(self):
        """
        .. py:method:: __repr__(self)

            Returns a string representation of the CiftiImg object.

            :return: A string representation of the CiftiImg object.
            :rtype: str
        """
        axes = "\n    ".join(map(repr, self.axis))
        return (
            "CiftiImg\n" f"    Dims: {len(self.shape)}\n" f"    -------\n" f"    {axes}"
        )

    @property
    def search(self):
        """.. py:method:: search()

        This method performs a search operation.

        Parameters
        ----------

        Returns
        -------
        None
            None

        """
        return

    @property
    def vertices(self) -> CiftiIndex:
        """.. py:method:: vertices()

        This method retrieves the vertices from each axis in the CiftiImg object. It iterates over each axis and attempts to access the `vertices` property. If the axis does not have a `vertices` property, it appends a `slice(None)` object to the result list. The method returns a tuple containing the vertices from each axis.

        Parameters
        ----------

        Returns
        -------
        CiftiIndex
            A tuple containing the vertices from each axis.

        """
        res = []
        for ax in self.axis:
            try:
                res.append(ax.vertices)
            except AttributeError:
                res.append(slice(None))
        return tuple(res)

    @property
    def voxels(self) -> CiftiIndex:
        """.. py:method:: voxels()

        This method returns a tuple containing the voxel indices for each axis in the CiftiImg object.

        Parameters
        ----------

        Returns
        -------
        CiftiIndex
            A tuple of voxel indices for each axis.

        """
        res = []
        for ax in self.axis:
            try:
                res.append(ax.voxels)
            except AttributeError:
                res.append(slice(None))
        return tuple(res)

    @property
    def axis(self):
        """.. py:method:: axis()

        This method builds a list of axes based on the input axes. It iterates through each axis in the input list and creates a new axis based on its type.

        Parameters
        ----------

        Returns
        -------
        list
            A list of new axes.

        """
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
        """.. py:method:: labels(self) -> LabelTable | None

        This method retrieves the label information from the Cifti image. It iterates through the axes of the Cifti image and checks if each axis is a LabelAxis. If a LabelAxis is found, it checks the length of the axis name. If the length is greater than 1, it returns None. Otherwise, it creates a LabelTable object using the first label table information from the axis and returns it.

        Parameters
        ----------

        Returns
        -------
        LabelTable|None
            The label table containing the name, label, meta, and data of the label table.

        """
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
        """.. py:method:: shape()

        Returns the shape of the `nibabel_obj`.

        Parameters
        ----------

        Returns
        -------
        tuple[int, ...]
            The shape of the `nibabel_obj`.

        """
        return self.nibabel_obj.shape

    def __getitem__(self, __index: CiftiIndex):
        """
        .. py:method:: __getitem__(self, __index: CiftiIndex)

            This method is used to retrieve data from the CiftiImg object based on the given index. It returns a new CiftiImg object with the indexed data.

            :param __index: The index used to retrieve the data. It can be a single index or a tuple of indexes.
            :type __index: CiftiIndex
            :return: A new CiftiImg object with the indexed data.
            :rtype: CiftiImg
        """
        # Get the data
        data = self.nibabel_obj.get_fdata()
        # Reformat __index
        if not isinstance(__index, tuple):
            __index = (__index,)
        __index = tuple(
            np.array(element).reshape(-1)
            if isinstance(element, Iterable) or isinstance(element, int)
            else element
            for element in __index
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
        new_axes = []
        if not isinstance(__index, tuple):  # or (isinstance(__index, np.ndarray)):
            __index = (__index,)
        for axis_idx, index_axis in enumerate(__index):
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
