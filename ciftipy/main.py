from __future__ import annotations
import numpy as np
import nibabel as nb
from nibabel.cifti2 import cifti2
from typing import Any, Mapping, Sequence, SupportsIndex, TypeAlias, TypeVar
# from typing_extensions import Ellipsis
import operator
import indexers
import yaml
from yaml.loader import SafeLoader
from thefuzz import fuzz
from thefuzz import process


DType = TypeVar("DType", bound=np.dtype[Any])
ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)
NDArray: TypeAlias = "np.ndarray[Any, np.dtype[ScalarType]]"
CiftiMaskTypes: TypeAlias = "NDArray[np.integer[Any]] | NDArray[np.bool_]"
CiftiMaskIndex: TypeAlias = "CiftiMaskTypes | tuple[CiftiMaskTypes, ...]"
CiftiBasicIndexTypes: TypeAlias = "SupportsIndex | slice | Ellipsis"
CiftiBasicIndex: TypeAlias = "CiftiBasicIndexTypes | tuple[CiftiBasicIndexTypes, ...]"
CiftiIndex: TypeAlias = "CiftiBasicIndex | CiftiMaskIndex"

with open('search_tokens.yaml') as f:
    search_tokens = yaml.load(f, Loader=SafeLoader)

for key in search_tokens.keys(): search_tokens[key] = set(search_tokens[key])

all_search_tokens = np.array(list(search_tokens['token_left'].union(hemi_tokens = search_tokens['token_right']).union(search_tokens['token_other'])))

np.char.replace(np.array(search_tokens['token_right']),'_RIGHT','')

class CiftiIndexHemi:
    def __init__(self,bm_axis: nb.cifti2.cifti2_axes.BrainModelAxis) -> None:
        self.size = bm_axis.size
        self.bm_structures = list(bm_axis.iter_structures())
        self.bm_structures_idxs = np.array(range(len(self.bm_structures)))
        self.bm_structures_names = np.array(list(map(operator.itemgetter(0), self.bm_structures)))
        self.bm_structures_name_idx_dict = dict(zip(self.bm_structures_names,self.bm_structures_idxs))
        self.bm_structures_slices = np.array(list(map(operator.itemgetter(1), self.bm_structures)))
        self.bm_axes = list(map(operator.itemgetter(2), self.bm_structures))
        

    def __getitem__(self, __index: str) -> CiftiIndex:
        # use fuzzer to get scores associated with either the 'left' or 'right' hemisphere
        scoresLR = list(map(operator.itemgetter(1),process.extract(__index,('left','right'),limit=None)))

        # checking fuzzer scores and getting set of standard cifti structures corresponding to hemispehere
        if scoresLR[0]>scoresLR[1]:
            hemi_tokens = search_tokens['token_left']
        elif scoresLR[0]<scoresLR[1]:
            hemi_tokens = search_tokens['token_right']
        else:
            hemi_tokens = search_tokens['token_other'] | search_tokens['token_left'] | search_tokens['token_right']
        
        # getting indicies associated with hemisphere brainstructures
        bm_indicies = operator.itemgetter(*set(self.bm_structures_names).intersection(hemi_tokens))(self.bm_structures_name_idx_dict)
        
        #indexing list of brainstructures for indexed structures
        new_bm_structures = operator.itemgetter(*bm_indicies)(self.bm_structures)

        #get verticies from brains structures
        mask = np.zeros(self.size) #Zeros
        slices = list(operator.itemgetter(1)(new_bm_structures)) #List of slices
        mask[np.r_[tuple(slices)]]=1     #get ranges from list of slices and then update mask
        
        return mask
    
class CiftiIndexStructure:
    def __init__(self,bm_axis: nb.cifti2.cifti2_axes.BrainModelAxis) -> None:
        self.size = bm_axis.size
        self.bm_structures = list(bm_axis.iter_structures())
        self.bm_structures_idxs = np.array(range(len(self.bm_structures)))
        self.bm_structures_names = np.array(list(map(operator.itemgetter(0), self.bm_structures)))
        self.bm_structures_name_idx_dict = dict(zip(self.bm_structures_names,self.bm_structures_idxs))
        self.bm_structures_slices = np.array(list(map(operator.itemgetter(1), self.bm_structures)))
        self.bm_axes = list(map(operator.itemgetter(2), self.bm_structures))   

    def __getitem__(self, __index: str) -> CiftiIndex:
        # use fuzzer to get scores associated with either the 'left' or 'right'
        # scoresLR = np.array(list(map(operator.itemgetter(1),process.extract(__index,('left','right'),limit=None))))

        scores_structure = np.array(list(map(operator.itemgetter(1),process.extract(__index, all_search_tokens,limit=None))))

        structure_bool_idx = scores_structure > 70

        valid_structure_set = set(all_search_tokens[structure_bool_idx])

        bm_indicies = operator.itemgetter(*set(self.bm_structures_names).intersection(valid_structure_set))(self.bm_structures_name_idx_dict)

        #indexing list of brainstructures for indexed structures
        new_bm_structures = operator.itemgetter(*bm_indicies)(self.bm_structures)

        #get verticies from brains structures
        mask = np.zeros(self.size) #Zeros
        slices = list(operator.itemgetter(1)(new_bm_structures)) #List of slices
        mask[np.r_[tuple(slices)]]=1     #get ranges from list of slices and then update mask
        
        return mask
        


        
    # check_continuity_list = lambda my_list: all(a+1==b for a, b in zip(my_list, my_list[1:]))

    # if isinstance(__index, list):
    #     assert check_continuity_list(__index) == True, 'Indicies must be continous.'



class CiftiSearch_struc:
    def __init__(self,bm_axis: nb.cifti2.cifti2_axes.BrainModelAxis) -> None:
        self.bm_structures = list(bm_axis.iter_structures())
        self.bm_structures_idxs = np.array(range(len(self.bm_structures)))
        self.bm_structures_names = np.array(list(map(operator.itemgetter(0), self.bm_structures)))
        self.bm_structures_slices = np.array(list(map(operator.itemgetter(1), self.bm_structures)))
        self.bm_axes = list(map(operator.itemgetter(2), self.bm_structures))

    def __getitem__(self, __index: str) -> CiftiIndex:
        process
        process.extract
        self.bm_structures_names

        
    # check_continuity_list = lambda my_list: all(a+1==b for a, b in zip(my_list, my_list[1:]))

    # if isinstance(__index, list):
    #     assert check_continuity_list(__index) == True, 'Indicies must be continous.'

    def __repr__(self) -> str: ...

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
    def label(self) -> CiftiIndexer: ...
    @property
    def key(self) -> CiftiIndexer: ...

class CiftiImg:
    def __init__(self, cifti: cifti2.Cifti2Image):
        self.nibabel_obj = cifti

    def __array__(self, dtype: DType = None):
        return self.nibabel_obj.get_fdata()
    
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
    def shape(self):
        return self.nibabel_obj.shape

    def __getitem__(self, __index: CiftiIndex):
        # Get the data
        data = self.nibabel_obj.get_fdata()
        # Index the dataobj
        new_data = data[__index]
        # Update the header. 
        if not isinstance(__index, tuple): #or (isinstance(__index, np.ndarray)):
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
        new_nb_obj =  nb.cifti2.Cifti2Image(new_data, self.nibabel_obj.header,
                                            self.nibabel_obj.nifti_header,
                                            self.nibabel_obj.extra,
                                            self.nibabel_obj.file_map)
        return CiftiImg(new_nb_obj)