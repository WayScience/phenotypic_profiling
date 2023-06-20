"""
utilities for validating logistic regression models on MitoCheck single-cell dataset
"""

import pathlib

import pandas as pd
import numpy as np
import seaborn as sns
from ccc.coef import ccc
from scipy.spatial.distance import squareform


def get_cell_health_corrs(
    final_profile_dataframe: pd.DataFrame, corr_type: str, cell_line: str
) -> pd.DataFrame:
    """
    get correlations between cell health indicators and classification profiles
    correlations are output in wide format

    Parameters
    ----------
    final_profile_dataframe : pd.DataFrame
        dataframe with cell health indicator and classification profiles matched by perturbation
    corr_type : str
        type of correlation ("pearson" or "ccc")
    cell_line : str
        cell line metadata to add to correlations

    Returns
    -------
    pd.DataFrame
        wide format dataframe with correlation info
    """

    # remove metadata columns
    cleaned_final_profile_dataframe = final_profile_dataframe.drop(
        columns=["Metadata_profile_id", "Metadata_pert_name", "Metadata_cell_line"]
    )

    # get correlations depending on type
    if corr_type == "pearson":
        corrs = cleaned_final_profile_dataframe.corr(method="pearson")
    if corr_type == "ccc":
        corrs = ccc(cleaned_final_profile_dataframe)
        corrs = squareform(corrs)
        np.fill_diagonal(corrs, 1.0)
        corrs = pd.DataFrame(
            corrs,
            index=cleaned_final_profile_dataframe.columns.tolist(),
            columns=cleaned_final_profile_dataframe.columns.tolist(),
        )

    # refactor correlations to easily readable format (from 70 cell health indicators)
    corrs = corrs.iloc[70:, :70]
    corrs.index.name = "phenotypic_class"
    corrs = corrs.reset_index()

    # add metadata to correlations
    corrs["cell_line"] = cell_line
    corrs["corr_type"] = corr_type

    return corrs


def get_tidy_long_corrs(final_profile_dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    get correlation values cell health indicators and classification profiles in tidy long format
    correlations are derived for each cell line and all cell lines
    pearson and ccc correlations are derived

    Parameters
    ----------
    final_profile_dataframe : pd.DataFrame
        dataframe with cell health indicator and classification profiles matched by perturbation

    Returns
    -------
    pd.DataFrame
        correlations in tidy long format
    """

    compiled_tidy_long_corrs = []

    # get correlations for all cell lines
    all_pearson_corrs = get_cell_health_corrs(final_profile_dataframe, "pearson", "all")
    all_ccc_corrs = get_cell_health_corrs(final_profile_dataframe, "ccc", "all")
    compiled_tidy_long_corrs += [all_pearson_corrs, all_ccc_corrs]

    for cell_line in final_profile_dataframe["Metadata_cell_line"].unique():
        # get subset of final profile dataframe for the particular cell line
        cell_line_profiles = final_profile_dataframe.loc[
            final_profile_dataframe["Metadata_cell_line"] == cell_line
        ]
        # get correlations for this cell line subset
        cell_line_pearson_corrs = get_cell_health_corrs(
            cell_line_profiles, "pearson", cell_line
        )
        cell_line_ccc_corrs = get_cell_health_corrs(
            cell_line_profiles, "ccc", cell_line
        )
        compiled_tidy_long_corrs += [cell_line_pearson_corrs, cell_line_ccc_corrs]

    # combine all correlations
    compiled_tidy_long_corrs = pd.concat(compiled_tidy_long_corrs)

    # convert wide dataframe to tidy long format
    compiled_tidy_long_corrs = pd.melt(
        compiled_tidy_long_corrs,
        id_vars=["phenotypic_class", "cell_line", "corr_type"],
        var_name="cell_health_indicator",
        value_name="corr_value",
    )

    return compiled_tidy_long_corrs


def get_corr_clustermap(
    tidy_corr_data: pd.DataFrame,
    cell_line: str,
    corr_type: str,
    model_type: str,
    feature_type: str,
):
    """
    get seaborn clustermap from tidy long correlation data
    will only plot data from the argument categories

    Parameters
    ----------
    tidy_corr_data : pd.DataFrame
        dataframe with correlation data in tidy long format
    cell_line : str
        all, A549, ES2, or HCC44
    corr_type : str
        pearson or ccc
    model_type : str
        final or shuffled_baseline
    feature_type : str
        CP, DP, or DP_and_DP
    """

    # get data of interest
    corr_data = tidy_corr_data.loc[
        (tidy_corr_data["cell_line"] == cell_line)
        & (tidy_corr_data["corr_type"] == corr_type)
        & (tidy_corr_data["model_type"] == model_type)
        & (tidy_corr_data["feature_type"] == feature_type)
    ]

    # drop rows that have "Negative" in phenotypic_class row
    # this will remove opposite values from single class models
    corr_data = corr_data[~corr_data["phenotypic_class"].str.contains("Negative")]

    # pivot the data to create matrix usable for graphing
    pivoted_corr_data = corr_data.pivot(
        "phenotypic_class", "cell_health_indicator", "corr_value"
    )

    # graph corr data
    sns.clustermap(
        pivoted_corr_data,
        xticklabels=pivoted_corr_data.columns,
        yticklabels=pivoted_corr_data.index,
        cmap="RdBu_r",
        linewidth=0.5,
        figsize=(20, 10),
    )
