import pandas as pd
from sklearn import model_selection

# TODO: click-ify with command line arguments.
def main():
    # TODO: parametrize
    data = pd.read_csv("../data/train.csv")
    data["fold_id"] = -1
    data = data.sample(frac=1).reset_index(drop=True)
    # TODO: parametrize?
    target = data.loc[:, "target"].values

    # TODO: parametrize
    n_splits = 5
    kf = model_selection.StratifiedKFold(n_splits=n_splits)

    for fold_id, (t_, v_) in enumerate(kf.split(X=data, y=target)):
        data.loc[v_, "fold_id"] = fold_id

    data.to_csv(f"../data/train_{n_splits}_folds.csv")


if __name__ == "__main__":
    main()
