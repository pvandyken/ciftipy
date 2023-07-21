from nibabel.cifti2 import cifti2_axes
from ciftipy.main import CiftiIndex1d

def index_brainmodel_axis(
    axis: cifti2_axes.BrainModelAxis, index: CiftiIndex1d
) -> cifti2_axes.BrainModelAxis: ...

def index_label_axis(
    axis: cifti2_axes.LabelAxis, index: CiftiIndex1d
) -> cifti2_axes.BrainModelAxis: ...

def index_parcel_axis(
    axis: cifti2_axes.ParcelsAxis, index: CiftiIndex1d
) -> cifti2_axes.ParcelsAxis: ...

def index_scalar_axis(
    axis: cifti2_axes.ScalarAxis, index: CiftiIndex1d
) -> cifti2_axes.ScalarAxis: ...

def index_series_axis(
    axis: cifti2_axes.SeriesAxis, index: CiftiIndex1d
) -> cifti2_axes.SeriesAxis: ...
