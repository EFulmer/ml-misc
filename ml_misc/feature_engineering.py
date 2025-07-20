import itertools

import pandas as pd


def categorical_features_pairwise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Create new categorical features consisting of all pairs of
    categorical features passed in as `cols`.

    Args:
        df: feature DataFrame.
        cols: names of columns that hold categorical features.

    Returns:
        DataFrame with one new column for each pairwise combination of
        categorical features
    """
    pairs = itertools.combinations(cols, 2)
    for feature_1, feature_2 in pairs:
        feature_name = f"{feature_1}_{feature_2}"
        new_feature = df.loc[:, feature_1].astype(str) + "_" + \
                df.loc[:, feature_2].astype(str)
        df.loc[:, feature_name] = new_feature
    return df
