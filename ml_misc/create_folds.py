from pathlib import Path

import pandas as pd
from sklearn import model_selection


# TODO: click-ify with command line arguments.
def main(input_file_name, n_splits, target_column):
    data = pd.read_csv((p := Path(input_file_name)))
    data["fold_id"] = -1
    data = data.sample(frac=1).reset_index(drop=True)

    target = data.loc[:, target_column].values

    kf = model_selection.StratifiedKFold(n_splits=n_splits)

    for fold_id, (t_, v_) in enumerate(kf.split(X=data, y=target)):
        data.loc[v_, "fold_id"] = fold_id

    output_file_name = p.parent / f"{p.stem}_{n_splits}_folds.csv"
    data.to_csv(output_file_name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
