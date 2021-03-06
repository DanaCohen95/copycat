import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from typing import Tuple
from sklearn.preprocessing import RobustScaler


def load_two_sigma_connect_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    p = "data/two-sigma-connect-rental-listing-inquiries/sigma_train_feat_0.01_tfidf_0.05.csv"
    df = pd.read_csv(p)
    X = df[['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price', ]
           + [col for col in df.columns if "feat" in col or "tfidf" in col]]
    y = df['interest_level'].apply(lambda s: 0 if s == "low" else 1 if s == "medium" else 2)
    return X, y


def load_costa_rica_dataset(plot_class_hist: bool = False
                            ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Costa Rica Poverty Prediction classification dataset.
    Choose specific features and balance the classes.

    Args:
        plot_class_hist: if True, create a plot of the class histogram before and after balancing.

    Returns:
        X: feature matrix dataframe [Samples X Features]
        y: class labels series [Samples]
    """
    # read csv
    p = r"data\costa-rican-household-poverty-prediction\train.csv"
    raw_df = pd.read_csv(p)

    # remove invalid features
    df = raw_df.drop(['Id', 'idhogar', 'dependency', 'edjefe', 'edjefa'], axis=1)
    # remove problematic features (rejected by pandas-profiling)
    df = df.drop(["SQBage", "SQBdependency", "SQBedjefe", "SQBescolari", "SQBhogar_nin", "SQBhogar_total",
                  "SQBmeaned", "SQBovercrowding", "age", "agesq", "elimbasu5", "escolari", "hhsize",
                  "hogar_mayor", "hogar_nin", "hogar_total", "qmobilephone",
                  "r4h1", "r4h2", "r4h3", "r4m1", "r4m2", "r4m3", "r4t1",
                  "rez_esc", "rez_esc", "tamhog", "tamviv", "v18q1", "v2a1", "meaneduc"], axis=1)

    # arbitrarily keep only the first 10 features lol
    X = df.drop(['Target'], axis=1).iloc[:, :]
    y = df['Target'] - 1

    # class 3 is VERY common, remove most of its samples.
    if plot_class_hist:
        y.hist()
        plt.title("Class Histogram")

    label_most_common = y.value_counts().idxmax()
    count_second_most_common = y.value_counts().sort_values().iloc[-2]

    idx_common = y.index.values[(y.values == label_most_common).nonzero()]
    np.random.RandomState(seed=42).shuffle(idx_common)
    idx_drop = idx_common[count_second_most_common:]

    X.drop(idx_drop, axis="rows", inplace=True)
    y.drop(idx_drop, axis="rows", inplace=True)

    if plot_class_hist:
        y.hist(width=0.2)
        plt.legend(["before balancing", "after balancing"], loc="best")
        plt.show(block=False)

    return X, y


def load_safe_drive_dataset():
    df = pd.read_csv("data/safe_driver/train.csv")
    df = df.drop(["ps_ind_05_cat", "ps_ind_14", "ps_ind_01", "ps_car_04_cat", "ps_car_09_cat", "ps_calc_12"], axis=1)

    X = df.drop(['target'], axis=1).iloc[:, :]
    y = df['target'].astype(int)

    label_most_common = y.value_counts().idxmax()
    count_second_most_common = y.value_counts().sort_values().iloc[-2]

    idx_common = y.index.values[(y.values == label_most_common).nonzero()]
    np.random.RandomState(seed=42).shuffle(idx_common)
    idx_drop = idx_common[count_second_most_common:]

    X.drop(idx_drop, axis="rows", inplace=True)
    y.drop(idx_drop, axis="rows", inplace=True)
    return X, y


def load_otto_dataset():
    df = pd.read_csv("data/otto_dataset/train.csv")
    X = df.drop(['target', 'id'], axis=1).iloc[:, :]
    y = df['target'].str.split('_', expand=True)[1].astype(int) - 1
    return X, y


def load_dataset(dataset_name):
    print("loading {} dataset".format(dataset_name))
    if dataset_name == 'costa_rica':
        return load_costa_rica_dataset()
    elif dataset_name == 'safe_drive':
        return load_safe_drive_dataset()
    elif dataset_name == 'otto':
        return load_otto_dataset()
    elif dataset_name == 'two_sigma_connect':
        return load_two_sigma_connect_dataset()
    else:
        raise ValueError("Unknown dataset: {}".format(dataset_name))


def over_sample_class(cls_inds, max_y):
    rs = np.random.RandomState(34)
    perm = rs.permutation(len(cls_inds))
    cls_inds = cls_inds[perm]

    num_reps = int(np.ceil(max_y / len(cls_inds)))
    _resample_inds = np.hstack([cls_inds] * num_reps)
    _resample_inds = _resample_inds[:max_y]
    return _resample_inds


def over_sample_for_class_balancing(X_train: pd.DataFrame,
                                    y_train: pd.Series
                                    ) -> Tuple[pd.DataFrame, pd.Series]:
    y_counts = y_train.value_counts()
    max_y = y_counts.max()

    resample_inds = []
    for cls in y_counts.index:
        cls_inds = np.nonzero((y_train == cls).values)[0]
        _resample_inds = over_sample_class(cls_inds, max_y)
        resample_inds.extend(_resample_inds)
    np.random.RandomState(34).shuffle(resample_inds)

    X_train_resample = X_train.iloc[resample_inds]
    y_train_resample = y_train.iloc[resample_inds]

    return X_train_resample, y_train_resample


def under_sample_for_class_balancing(X_train: pd.DataFrame,
                                     y_train: pd.Series
                                     ) -> Tuple[pd.DataFrame, pd.Series]:
    y_counts = y_train.value_counts()
    min_y = y_counts.min()

    keep_inds = []
    for cls in y_counts.index:
        cls_inds = np.nonzero((y_train == cls).values)[0]
        perm = np.random.RandomState(34).permutation(len(cls_inds))
        _keep_inds = cls_inds[perm[:min_y]]
        keep_inds.extend(_keep_inds)
    np.random.RandomState(34).shuffle(keep_inds)

    X_train_clip = X_train.iloc[keep_inds]
    y_train_clip = y_train.iloc[keep_inds]

    return X_train_clip, y_train_clip


def prepare_data(X: pd.DataFrame,
                 y: pd.Series,
                 num_samples_to_keep: int = None,
                 class_balancing_strategy: str = None,
                 normalize_features: bool = False
                 ) -> Tuple[int, int, int,
                            pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
                            np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray]:
    """
    Split data to train set and validation set.
    Calculate data measures.

    Args:
        X: feature matrix dataframe [Samples X Features]
        y: class labels series [Samples]
        num_samples_to_keep: randomly keep only some of the samples

    Returns:
        n_samples, n_features, n_classes: as the name implies
        X_train, X_valid, y_train, y_valid: train-validation split of the data
        y_train_onehot, y_valid_onehot: labels as one-hot matrices instead of integer vectors
        class_weights: multiply the loss by these values to give more weight to misrepresented classes.
                       let C be a vector of class counts in the training set (length n_classes), then
                       class_weights = (1 / C) / sum(1/C) * n_classes
    """
    n_samples, n_features = X.shape
    n_classes = len(y.unique())

    if num_samples_to_keep is not None:
        num_samples_to_keep = min(num_samples_to_keep, len(X) - n_classes)
        num_extra = len(X) - num_samples_to_keep
        X, _, y, _ = train_test_split(X, y,
                                      train_size=num_samples_to_keep, test_size=num_extra,
                                      shuffle=True, stratify=y.values, random_state=34)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=34,
                                                          shuffle=True, stratify=y.values)
    if class_balancing_strategy == "over_sample":
        X_train, y_train = over_sample_for_class_balancing(X_train, y_train)
    elif class_balancing_strategy == "under_sample":
        X_train, y_train = under_sample_for_class_balancing(X_train, y_train)

    if normalize_features:
        scaler = RobustScaler().fit(X_train)
        X_train_np = scaler.transform(X_train)
        X_train = pd.DataFrame(data=X_train_np, columns=X_train.columns)
        X_valid_np = scaler.transform(X_valid)
        X_valid = pd.DataFrame(data=X_valid_np, columns=X_valid.columns)

    y_train_onehot = to_categorical(y_train, num_classes=n_classes)
    y_valid_onehot = to_categorical(y_valid, num_classes=n_classes)
    y_onehot = to_categorical(y)

    class_counts = y_train.value_counts().sort_index().values
    class_weights = 1. / class_counts
    class_weights /= class_weights.sum() / float(n_classes)
    class_weights = class_weights.reshape((1, -1, 1))
    class_weights = class_weights.astype(np.float32)

    return (n_samples, n_features, n_classes,
            X_train, X_valid, y_train, y_valid,
            y_train_onehot, y_valid_onehot, y_onehot,
            class_weights)
