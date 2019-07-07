import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


def load_costa_rica_dataset():
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
                  "rez_esc", "rez_esc", "tamhog", "tamviv", "v18q1", "v2a1"], axis=1)
    # arbitrarily keep only the first 10 features lol
    X = df.drop(['Target'], axis=1).iloc[:, :10]
    y = df['Target'] - 1

    # class 3 is VERY common, remove most of its samples.
    y.hist()
    plt.title("Class Histogram")

    label_most_common = y.value_counts().idxmax()
    count_second_most_common = y.value_counts().sort_values().iloc[-2]

    idx_common, = (y == label_most_common).nonzero()
    np.random.shuffle(idx_common)
    idx_drop = idx_common[count_second_most_common:]

    X.drop(idx_drop, axis="rows", inplace=True)
    y.drop(idx_drop, axis="rows", inplace=True)

    y.hist(width=0.2)
    plt.legend(["before balancing", "after balancing"], loc="best")
    plt.show(block=False)

    return X, y


def prepare_data(X, y):
    n_samples, n_features = X.shape
    n_classes = len(y.unique())

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=7)

    y_train_onehot = to_categorical(y_train)
    y_valid_onehot = to_categorical(y_valid)

    return (n_samples, n_features, n_classes,
            X_train, X_valid, y_train, y_valid,
            y_train_onehot, y_valid_onehot)


if __name__ == '__main__':
    load_costa_rica_dataset()
