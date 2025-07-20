from typing import Optional

import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing


def train_cv(
    df: pd.DataFrame,
    fold_id: int,
    non_vars: list[str],
    num_vars: list[str],
    cat_vars: list[str],
    target_var: str,
    model_class: type,
    model_hyperparams: dict,
    target_mapping: Optional[dict[str, int]]=None,
) -> float:
    """Run a categorical model.

    Args:
        non_vars: list of column names that are not to be used as
            features/variables.
        target_mapping:

    Returns:
        AUC
    """
    try:
        df = df.drop(non_vars, axis=1)
    except Exception as e:
        pass

    if target_mapping is not None:
        df.loc[:, target_var] = df.loc[:, target_var].map(target_mapping)

    for var in cat_vars:
        enc = preprocessing.OrdinalEncoder()
        enc.fit(df.loc[:, var].values.reshape(-1, 1))
        # TODO: this typecasting raises warnings, at least in Jupyter
        df.loc[:, var] = enc.transform(df.loc[:, var].values.reshape(-1, 1))


    vars_ = list(set(num_vars) | set(cat_vars))
    df_train = df[df.fold_id != fold_id].reset_index(drop=True)
    df_valid = df[df.fold_id == fold_id].reset_index(drop=True)

    X_train = df_train.loc[:, vars_].values
    X_valid = df_valid.loc[:, vars_].values
    y_train = df_train.loc[:, target_var].values.astype(np.int64)
    y_valid = df_valid.loc[:, target_var].values.astype(np.int64)

    model = model_class(**model_hyperparams)
    model.fit(X_train, y_train)

    valid_preds = model.predict_proba(X_valid)[:, 1]
    auc = metrics.roc_auc_score(y_valid, valid_preds)
    return auc
