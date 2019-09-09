from models import get_student_nn_classifier, get_vanilla_nn_classifier
from models import get_student2sigma, get_vanilla_2sigma
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
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
from tqdm import tqdm
import time


def calculate_scores(y: Union[np.ndarray, pd.Series],
                     probs: np.ndarray
                     ) -> Dict[str, np.ndarray]:
    if np.ndim(y) == 2:
        y = np.argmax(y, axis=1)
    # n_classes = probs.shape[1]
    preds = np.argmax(probs, axis=1)

    scores = {
        "accuracy": accuracy_score(y, preds),
        # "average_precision_macro": average_precision_score(
        #     to_categorical(y, num_classes=n_classes), probs, average="macro"),
        "f1_macro": f1_score(y, preds, average="macro")
    }
    return scores


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
                                  xgb_learning_rate: float,
                                  data_preparation_func: Callable
                                  ) -> Dict[str, np.ndarray]:
    # split data
    X_train, X_test = X[inds_train], X[inds_test]
    y_cls_train, y_cls_test = y[inds_train], y[inds_test]

    if data_preparation_func is not None:
        X_train, y_cls_train, X_test = data_preparation_func(X_train, y_cls_train, X_test)

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
    # model = get_student2sigma(n_classes, n_features, n_shap_features, expected_logits, print_summary=False)

    model.fit(X_train, [y_cls_train_onehot, y_shap_train], epochs=epochs, verbose=0)

    # evaluate model
    probs_test, shaps_test = model.predict(X_test)
    scores_test = calculate_scores(y_cls_test, probs_test)

    probs_train, shaps_train = model.predict(X_train)
    scores_train = calculate_scores(y_cls_train, probs_train)
    scores_train = {"train_" + k: v for k, v in scores_train.items()}

    scores = scores_test
    scores.update(scores_train)
    return scores


def train_and_evaluate_xgboost(X: np.ndarray,
                               y: Tuple[np.ndarray, np.ndarray],
                               inds_train: np.ndarray,
                               inds_test: np.ndarray,
                               max_depth: int,
                               n_estimators: int,
                               learning_rate: float,
                               data_preparation_func: Callable
                               ) -> Dict[str, np.ndarray]:
    # split data
    X_train, X_test = X[inds_train], X[inds_test]
    y_train, y_test = y[inds_train], y[inds_test]

    if data_preparation_func is not None:
        X_train, y_train, X_test = data_preparation_func(X_train, y_train, X_test)

    # build and train model
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    xgb_model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate,
                              objective="multi:softmax")
    xgb_model.fit(X_train, y_train)

    # evaluate model
    probs_test = xgb_model.predict_proba(X_test)
    scores_test = calculate_scores(y_test, probs_test)

    probs_train = xgb_model.predict_proba(X_train)
    scores_train = calculate_scores(y_train, probs_train)
    scores_train = {"train_" + k: v for k, v in scores_train.items()}

    scores = scores_test
    scores.update(scores_train)
    return scores


def train_and_evaluate_vanilla_nn(X: np.ndarray,
                                  y: Tuple[np.ndarray, np.ndarray],
                                  inds_train: np.ndarray,
                                  inds_test: np.ndarray,
                                  n_classes: int,
                                  n_features: int,
                                  epochs: int,
                                  data_preparation_func: Callable
                                  ) -> Dict[str, np.ndarray]:
    # split data
    X_train, X_test = X[inds_train], X[inds_test]
    y_train, y_test = y[inds_train], y[inds_test]

    if data_preparation_func is not None:
        X_train, y_train, X_test = data_preparation_func(X_train, y_train, X_test)

    y_train_onehot = to_categorical(y_train, num_classes=n_classes)

    # build and train model
    model = get_vanilla_nn_classifier(n_classes, n_features, print_summary=False)
    # model = get_vanilla_2sigma(n_classes, n_features, print_summary=False)

    model.fit(X_train, y_train_onehot, epochs=epochs, verbose=0)

    # evaluate model
    probs_test = model.predict(X_test)
    scores = calculate_scores(y_test, probs_test)

    # evaluate model
    probs_test = model.predict(X_test)
    scores_test = calculate_scores(y_test, probs_test)

    probs_train = model.predict(X_train)
    scores_train = calculate_scores(y_train, probs_train)
    scores_train = {"train_" + k: v for k, v in scores_train.items()}

    scores = scores_test
    scores.update(scores_train)
    return scores


def generic_train_and_evaluate(args_tuple: Tuple
                               ) -> Dict[str, np.ndarray]:
    # print("pid {pid}: started train_and_evaluate".format(pid=os.getpid()))
    X, y, inds_train, inds_test, train_and_evaluate_fn, fn_kwargs = args_tuple
    scores = train_and_evaluate_fn(X, y, inds_train, inds_test, **fn_kwargs)
    # print("pid {pid}: finished train_and_evaluate".format(pid=os.getpid()))
    return scores


def cross_validation(X: np.ndarray,
                     y: np.ndarray,
                     train_and_evaluate_fn: Callable,
                     fn_kwargs: Dict[str, Any] = None,
                     n_splits: int = 10,
                     test_size: int = 0.2,
                     multiprocess: bool = True,
                     tqdm_description: str = None,
                     ) -> pd.DataFrame:
    if multiprocess:
        pool = Pool(3)
        map_func = pool.imap
    else:
        map_func = map

    if fn_kwargs is None:
        fn_kwargs = {}

    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    folds = list(cv.split(np.zeros(X.shape[0]), y))
    packed_args_list = [(X, y, inds_train, inds_test, train_and_evaluate_fn, fn_kwargs)
                        for inds_train, inds_test in folds]
    all_scores = list(map_func(generic_train_and_evaluate, tqdm(packed_args_list, desc=tqdm_description)))
    scores_df = pd.DataFrame(all_scores)
    return scores_df


def example_cross_validation() -> None:
    from data_utils import load_dataset, prepare_two_sigma_connect_data
    now_str = time.strftime("%Y-%m-%d-%H-%M-%S")

    # dataset_name = "costa_rica"
    # data_preparation_func = None

    dataset_name = "two_sigma_connect"
    data_preparation_func = prepare_two_sigma_connect_data

    X, y = load_dataset(dataset_name)
    n_samples, n_features = X.shape
    n_classes = len(y.unique())

    multiprocess = True
    n_splits = 100
    # for model_type in ["xgboost", "student_nn", "vanilla_nn"]:
    for model_type in ["xgboost", "vanilla_nn", "student_nn"]:
    # for model_type in ["student_nn"]:

        if model_type == "vanilla_nn":
            fn_kwargs = {
                "n_classes": n_classes,
                "n_features": n_features,
                "epochs": 15,
                "data_preparation_func": data_preparation_func
            }
            train_and_evaluate_fn = train_and_evaluate_vanilla_nn

        elif model_type == "xgboost":
            fn_kwargs = {
                "max_depth": 10,
                "n_estimators": 30,
                "learning_rate": 0.1,
                "data_preparation_func": data_preparation_func
            }
            train_and_evaluate_fn = train_and_evaluate_xgboost

        elif model_type == "student_nn":
            fn_kwargs = {
                "n_classes": n_classes,
                "n_features": n_features,
                "n_shap_features": 5,
                "epochs": 15,
                "xgb_max_depth": 10,
                "xgb_n_estimators": 30,
                "xgb_learning_rate": 0.1,
                "data_preparation_func": data_preparation_func
            }
            train_and_evaluate_fn = train_and_evaluate_student_nn

        else:
            raise ValueError("Unsupported model type: " + model_type)

        scores_df = cross_validation(X=X.values,
                                     y=y.values,
                                     train_and_evaluate_fn=train_and_evaluate_fn,
                                     fn_kwargs=fn_kwargs,
                                     n_splits=n_splits,
                                     multiprocess=multiprocess,
                                     tqdm_description=model_type)

        print(scores_df)

        results_dir = osp.join("results", now_str, model_type)
        os.makedirs(results_dir, exist_ok=True)

        scores_path = osp.join(results_dir, "scores.csv")
        scores_df.to_csv(scores_path)

        fn_kwargs_path = osp.join(results_dir, "fn_kwargs.json")
        fn_kwargs_to_dump = {k: v for k, v in fn_kwargs.items() if k != "data_preparation_func"}
        with open(fn_kwargs_path, 'w') as f:
            json.dump(fn_kwargs_to_dump, f, indent=2)


if __name__ == '__main__':
    example_cross_validation()
