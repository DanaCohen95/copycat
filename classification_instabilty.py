from models import get_student_nn_classifier, get_vanilla_nn_classifier
import scipy.spatial as sp
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, cohen_kappa_score
from tensorflow.keras.utils import to_categorical
import pandas as pd
from multiprocessing import Pool
from typing import Union, Tuple, Dict, Callable, Any
import os
import os.path as osp
from xgboost import XGBClassifier
from xgboost_utils import calculate_shap_values
import warnings
import json
import matplotlib.pyplot as plt
import time


def calculate_disagreement(probs1: np.ndarray,
                           probs2: np.ndarray
                           ) -> int:
    preds1 = np.argmax(probs1, axis=1)
    preds2 = np.argmax(probs2, axis=1)

    return np.sum(np.abs(preds1 - preds2) != 0) / preds1.size


def calculate_jsd(probs1: np.ndarray,
                  probs2: np.ndarray
                  ) -> int:
    return np.average(sp.distance.jensenshannon(probs1.transpose(), probs2.transpose()))

def calculate_cks(probs1: np.ndarray,
                  probs2: np.ndarray
                  ) -> int:
    preds1 = np.argmax(probs1, axis=1)
    preds2 = np.argmax(probs2, axis=1)
    return cohen_kappa_score(preds1, preds2)


def train_and_evaluate_student_nn(X: np.ndarray,
                                  y: Tuple[np.ndarray, np.ndarray],
                                  inds_train: np.ndarray,
                                  inds_test: np.ndarray,
                                  n_classes: int,
                                  n_features: int,
                                  n_shap_features: int,
                                  epochs: int,
                                  xgb_max_depth: int,
                                  xgb_n_estimators: int,
                                  xgb_learning_rate: float
                                  ) -> Dict[str, np.ndarray]:
    # split data
    X_trains, X_test = (X[inds_train[:int(inds_train.size / 2)]],
                        X[inds_train[int(inds_train.size / 2):]]), X[inds_test]
    y_cls_trains, y_cls_test = (y[inds_train[:int(inds_train.size / 2)]],
                                y[inds_train[int(inds_train.size / 2):]]), y[inds_test]

    probs = []
    for i in range(2):
        X_train = X_trains[i]
        y_cls_train = y_cls_trains[i]
        y_cls_train_onehot = to_categorical(y_cls_train, num_classes=n_classes)

        # build and train XGBoost model, calculate shap values
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        xgb_model = XGBClassifier(max_depth=xgb_max_depth,
                                  n_estimators=xgb_n_estimators,
                                  learning_rate=xgb_learning_rate,
                                  objective="multi:softmax")
        xgb_model.fit(X_train, y_cls_train)
        y_shap_train, expected_logits = calculate_shap_values(
            xgb_model, X_train, n_shap_features)

        # build and train model
        model = get_student_nn_classifier(n_classes, n_features, n_shap_features, expected_logits, print_summary=False)
        model.fit(X_train, [y_cls_train_onehot, y_shap_train], epochs=epochs, verbose=0)

        # evaluate model
        temp, _ = model.predict(X_test)
        probs.append(temp)

    pOfDisagreement = calculate_disagreement(probs[0], probs[1])
    JSD = calculate_jsd(probs[0], probs[1])
    CKS = calculate_cks(probs[0], probs[1])

    scores = {
        "Percentage of disagreement": pOfDisagreement,
        "Jensen-Shannon divergence": JSD,
        "Cohen-Kappa score": CKS
    }
    return scores


def train_and_evaluate_xgboost(X: np.ndarray,
                               y: Tuple[np.ndarray, np.ndarray],
                               inds_train: np.ndarray,
                               inds_test: np.ndarray,
                               max_depth: int = 5,
                               n_estimators: int = 30,
                               learning_rate: float = 0.1
                               ) -> Dict[str, np.ndarray]:
    # split data
    X_trains, X_test = (X[inds_train[:int(inds_train.size / 2)]],
                        X[inds_train[int(inds_train.size / 2):]]), X[inds_test]
    y_trains, y_test = (y[inds_train[:int(inds_train.size / 2)]],
                                y[inds_train[int(inds_train.size / 2):]]), y[inds_test]

    probs = []
    for i in range(2):
        X_train = X_trains[i]
        y_train = y_trains[i]
        # build and train model
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        xgb_model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate,
                              objective="multi:softmax")
        xgb_model.fit(X_train, y_train)

        # evaluate model
        probs.append(xgb_model.predict_proba(X_test))

    pOfDisagreement = calculate_disagreement(probs[0], probs[1])
    JSD = calculate_jsd(probs[0], probs[1])
    CKS = calculate_cks(probs[0], probs[1])

    scores = {
        "Percentage of disagreement": pOfDisagreement,
        "Jensen-Shannon divergence": JSD,
        "Cohen-Kappa score": CKS
    }
    return scores


def train_and_evaluate_vanilla_nn(X: np.ndarray,
                                  y: Tuple[np.ndarray, np.ndarray],
                                  inds_train: np.ndarray,
                                  inds_test: np.ndarray,
                                  n_classes: int,
                                  n_features: int,
                                  epochs: int
                                  ) -> Dict[str, np.ndarray]:

    # split data
    X_trains, X_test = (X[inds_train[:int(inds_train.size / 2)]],
                        X[inds_train[int(inds_train.size / 2):]]), X[inds_test]
    y_trains, y_test = (y[inds_train[:int(inds_train.size / 2)]],
                                y[inds_train[int(inds_train.size / 2):]]), y[inds_test]

    probs = []
    for i in range(2):
        X_train = X_trains[i]
        y_train = y_trains[i]
        y_train_onehot = to_categorical(y_train, num_classes=n_classes)

        # build and train model
        model = get_vanilla_nn_classifier(n_classes, n_features, print_summary=False)
        model.fit(X_train, y_train_onehot, epochs=epochs, verbose=0)

        # evaluate model
        probs.append(model.predict(X_test))

    pOfDisagreement = calculate_disagreement(probs[0], probs[1])
    JSD = calculate_jsd(probs[0], probs[1])
    CKS = calculate_cks(probs[0], probs[1])

    scores = {
        "Percentage of disagreement": pOfDisagreement,
        "Jensen-Shannon divergence": JSD,
        "Cohen-Kappa score": CKS
    }
    return scores


def generic_train_and_evaluate(args_tuple: Tuple
                               ) -> Dict[str, np.ndarray]:
    print("pid {pid}: started train_and_evaluate".format(pid=os.getpid()))
    X, y, inds_train, inds_test, train_and_evaluate_fn, fn_kwargs = args_tuple
    scores = train_and_evaluate_fn(X, y, inds_train, inds_test, **fn_kwargs)
    print("pid {pid}: finished train_and_evaluate".format(pid=os.getpid()))
    return scores


def check_stability(X: np.ndarray,
                    y: np.ndarray,
                    train_and_evaluate_fn: Callable,
                    fn_kwargs: Dict[str, Any] = None,
                    n_splits: int = 10,
                    test_size: int = 0.1,
                    multiprocess: bool = True
                    ) -> pd.DataFrame:
    if multiprocess:
        pool = Pool(3)
        map_func = pool.map
    else:
        map_func = map

    if fn_kwargs is None:
        fn_kwargs = {}

    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    folds = list(cv.split(np.zeros(X.shape[0]), y))
    packed_args_list = [(X, y, inds_train, inds_test, train_and_evaluate_fn, fn_kwargs)
                        for inds_train, inds_test in folds]
    all_scores = list(map_func(generic_train_and_evaluate, packed_args_list))
    scores_df = pd.DataFrame(all_scores)
    return scores_df


def run_all() -> None:
    from data_utils import load_costa_rica_dataset
    now_str = time.strftime("%Y-%m-%d-%H-%M-%S")

    X, y = load_costa_rica_dataset()
    n_samples, n_features = X.shape
    n_classes = len(y.unique())

    n_splits = 100
    for model_type in ["student_nn", "vanilla_nn"]:

        if model_type == "vanilla_nn":
            fn_kwargs = {
                "n_classes": n_classes,
                "n_features": n_features,
                "epochs": 50
            }
            train_and_evaluate_fn = train_and_evaluate_vanilla_nn

        elif model_type == "xgboost":
            fn_kwargs = {
                "max_depth": 10,
                "n_estimators": 100,
                "learning_rate": 0.1
            }
            train_and_evaluate_fn = train_and_evaluate_xgboost

        elif model_type == "student_nn":
            fn_kwargs = {
                "n_classes": n_classes,
                "n_features": n_features,
                "n_shap_features": 10,
                "epochs": 10,
                "xgb_max_depth": 10,
                "xgb_n_estimators": 30,
                "xgb_learning_rate": 0.1
            }
            train_and_evaluate_fn = train_and_evaluate_student_nn

        else:
            raise ValueError("Unsupported model type: " + model_type)

        scores_df = check_stability(X=X.values,
                                    y=y.values,
                                    train_and_evaluate_fn=train_and_evaluate_fn,
                                    fn_kwargs=fn_kwargs,
                                    n_splits=n_splits,
                                    multiprocess=True)

        print(scores_df)

        results_dir = "stability_results/" + now_str + "_" + model_type
        os.makedirs(results_dir, exist_ok=True)

        scores_path = osp.join(results_dir, "scores.csv")
        scores_df.to_csv(scores_path)

        fn_kwargs_path = osp.join(results_dir, "fn_kwargs.json")
        with open(fn_kwargs_path, 'w') as f:
            json.dump(fn_kwargs, f, indent=2)

            # hist_path = osp.join(results_dir, "scores_hist.png")
            # plt.figure()
            # scores_df.hist()
            # plt.savefig(hist_path)


if __name__ == '__main__':
    run_all()
