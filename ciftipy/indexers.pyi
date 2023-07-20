from nibabel.cifti2 import cifti2_axes
from ciftipy.main import CiftiIndex1d

def index_brainmodel_axis(
    axis: cifti2_axes.BrainModelAxis, index: CiftiIndex1d
) -> cifti2_axes.BrainModelAxis: ...

def index_label_axis(
    axis: cifti2_axes.LabelAxis, index: CiftiIndex1d
) -> cifti2_axes.BrainModelAxis: ...