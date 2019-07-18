from models import get_student_nn_classifier, get_vanilla_nn_classifier
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
from tensorflow.keras.utils import to_categorical
import pandas as pd
from multiprocessing import Pool
from typing import Union, Tuple, Dict, Callable, Any
import os


def calculate_scores(y: Union[np.ndarray, pd.Series],
                     probs: np.ndarray
                     ) -> Dict[str, np.ndarray]:
    if np.ndim(y) == 2:
        y = np.argmax(y, axis=1)
    preds = np.argmax(probs, axis=1)

    scores = {
        "accuracy": accuracy_score(y, preds),
        "average_precision_macro": average_precision_score(to_categorical(y), probs, average="macro"),
        "f1_macro": f1_score(y, preds, average="macro")
    }
    return scores


def train_and_evaluate_student_nn(X: np.ndarray,
                                  y: Tuple[np.ndarray, np.ndarray],
                                  inds_train: np.ndarray,
                                  inds_test: np.ndarray,
                                  n_classes: int,
                                  n_features: int,
                                  expected_logits: np.ndarray,
                                  class_weights: np.ndarray,
                                  epochs: int
                                  ) -> Dict[str, np.ndarray]:
    # split data
    y_cls, y_shap = y

    X_train, X_test = X[inds_train], X[inds_test]
    y_cls_train, y_cls_test = y_cls[inds_train], y_cls[inds_test]
    y_shap_train, y_shap_test = y_shap[inds_train], y_shap[inds_test]

    # build and train model
    model = get_student_nn_classifier(n_classes, n_features, expected_logits,
                                      class_weights=class_weights, print_summary=False)
    model.fit(X_train, [y_cls_train, y_shap_train], epochs=epochs, verbose=0)

    # evaluate model
    probs_test, shaps_test = model.predict(X_test)
    scores = calculate_scores(y_cls_test, probs_test)

    return scores


def generic_train_and_evaluate_fn(args_tuple: Tuple
                                  ) -> Dict[str, np.ndarray]:
    print("pid {pid}: started train_and_evaluate".format(pid=os.getpid()))
    X, y, inds_train, inds_test, train_and_evaluate_fn, kwargs = args_tuple
    scores = train_and_evaluate_fn(X, y, inds_train, inds_test, **kwargs)
    print("pid {pid}: finished train_and_evaluate".format(pid=os.getpid()))
    return scores


def cross_validation(X: np.ndarray,
                     y: Union[np.ndarray, pd.Series, Tuple[np.ndarray, np.ndarray]],
                     y_stratify: Union[np.ndarray, pd.Series],
                     train_and_evaluate_fn: Callable,
                     fn_kwargs: Dict[str, Any] = None,
                     n_splits: int = 10,
                     test_size: int = 0.2,
                     multiprocess: bool = True
                     ) -> pd.DataFrame:
    if multiprocess:
        pool = Pool()
        map_func = pool.map
    else:
        map_func = map

    if fn_kwargs is None:
        fn_kwargs = {}

    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    folds = list(cv.split(np.zeros(X.shape[0]), y_stratify))
    packed_args_list = [(X, y, inds_train, inds_test, train_and_evaluate_fn, fn_kwargs)
                        for inds_train, inds_test in folds]
    all_scores = list(map_func(generic_train_and_evaluate_fn, packed_args_list))
    scores_df = pd.DataFrame(all_scores)
    return scores_df


def example_cross_validation() -> None:
    from data_utils import load_costa_rica_dataset, prepare_data
    from xgboost_utils import fit_xgboost_classifier, calculate_shap_values

    use_weighted_shap_loss = False
    model_type = "student"
    assert model_type in ["student", "vanilla"]

    X, y = load_costa_rica_dataset()
    (n_samples, n_features, n_classes,
     X_train, X_valid, y_train, y_valid,
     y_train_onehot, y_valid_onehot,
     class_weights) = prepare_data(X, y)

    if not use_weighted_shap_loss:
        class_weights = None

    xgb_model = fit_xgboost_classifier(X_train, y_train)
    shap_values_train, expected_logits = calculate_shap_values(xgb_model, X_train)
    shap_values_valid, _ = calculate_shap_values(xgb_model, X_valid)

    fn_kwargs = {
        "n_classes": n_classes,
        "n_features": n_features,
        "expected_logits": expected_logits,
        "class_weights": class_weights,
        "epochs": 5
    }

    scores_df = cross_validation(X=X_train.values,
                                 y=(y_train_onehot, shap_values_train),
                                 y_stratify=y_train.values,
                                 train_and_evaluate_fn=train_and_evaluate_student_nn,
                                 fn_kwargs=fn_kwargs,
                                 n_splits=2,
                                 multiprocess=True)

    print(scores_df)
    pass


if __name__ == '__main__':
    example_cross_validation()
