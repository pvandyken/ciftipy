import re
from nibabel.cifti2 import cifti2_axes
from tests import strategies as cp_st
from hypothesis import strategies as st, given
from tests import structures
from ciftipy.main import CiftiSearch
import numpy as np


@given(
    axis=cp_st.brain_model_axes(names=st.sampled_from(structures.STRUCTURES_LEFT)),
    query=st.from_regex(re.compile(r"r(ight)?", flags=re.IGNORECASE)),
)
def test_all_false_when_missing_hemisphere_selected(
    axis: cifti2_axes.BrainModelAxis, query: str
):

    """.. py:function:: test_all_false_when_missing_hemisphere_selected(axis: cifti2_axes.BrainModelAxis, query: str)
    
    This function tests whether all values in the search result are False when a hemisphere is missing in the selected axis.

    Parameters
    ----------
    axis : cifti2_axes.BrainModelAxis
        The brain model axis.
    query : str
        The query string.

    Returns
    -------
    None
        None

    """
    search = CiftiSearch(axis)
    assert np.all(~search[query])


@given(
    axis=cp_st.brain_model_axes(names=st.sampled_from(structures.STRUCTURES_RIGHT)),
    query=st.from_regex(re.compile(r"r(ight)?", flags=re.IGNORECASE)),
)
def test_all_true_when_only_hemisphere_selected(
    axis: cifti2_axes.BrainModelAxis, query: str
):

    """.. py:function:: test_all_true_when_only_hemisphere_selected(axis: cifti2_axes.BrainModelAxis, query: str)
    
    Test function to check if all values in the search result are True when only the hemisphere is selected.

    Parameters
    ----------
    axis : cifti2_axes.BrainModelAxis
        The brain model axis.
    query : str
        The query string.

    Returns
    -------
    None
        None

    """
    search = CiftiSearch(axis)
    assert np.all(search[query])


@given(
    axis=cp_st.brain_model_axes(names=st.sampled_from(structures.STRUCTURES_CORTEX)),
    query=st.from_regex(re.compile(r"cortex", flags=re.IGNORECASE)),
)
def test_all_true_when_only_cortex_models_and_cortex_selected(
    axis: cifti2_axes.BrainModelAxis, query: str
):

    """.. py:function:: test_all_true_when_only_cortex_models_and_cortex_selected(axis: cifti2_axes.BrainModelAxis, query: str)
    
    This function tests whether all values in the search result for the given query are True when only cortex models are selected in the brain model axis.

    Parameters
    ----------
    axis : cifti2_axes.BrainModelAxis
        The brain model axis containing the selected models.
    query : str
        The query string to search for.

    Returns
    -------
    None
        None

    """
    search = CiftiSearch(axis)
    assert np.all(search[query])


@given(
    axis=cp_st.brain_model_axes(
        names=st.sampled_from(
            list(set(structures.STRUCTURES) - set(structures.STRUCTURES_CORTEX))
        )
    ),
    query=st.from_regex(re.compile(r"cortex", flags=re.IGNORECASE)),
)
def test_all_false_when_no_cortex_models_and_cortex_selected(
    axis: cifti2_axes.BrainModelAxis, query: str
):

    """.. py:function:: test_all_false_when_no_cortex_models_and_cortex_selected(axis: cifti2_axes.BrainModelAxis, query: str)
    
    Test function to check if all values in the search result are False when no cortex models are present and the query is related to the cortex.

    Parameters
    ----------
    axis : cifti2_axes.BrainModelAxis
        The brain model axis.
    query : str
        The query string.

    Returns
    -------
    None
        None

    """
    search = CiftiSearch(axis)
    assert np.all(search[query])
