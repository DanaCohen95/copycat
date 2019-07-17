import warnings
from xgboost import XGBClassifier
import shap
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from typing import Tuple


def fit_xgboost_classifier(X: pd.DataFrame,
                           y: pd.Series
                           ) -> XGBClassifier:
    """ create & fit an XGBoost tree-booster classifier for multiclass data """
    warnings.filterwarnisngs("ignore", category=DeprecationWarning)
    xgb_model = XGBClassifier(max_depth=5, n_estimators=30, learning_rate=0.1, objective="multi:softmax")
    xgb_model.fit(X, y)
    return xgb_model


def evaluate_xgboost_classifier(xgb_model: XGBClassifier,
                                X: pd.DataFrame,
                                y: pd.Series
                                ) -> None:
    """ evaluate an XGBoost classifier and print a classification report """
    preds = xgb_model.predict(X)
    print(classification_report(y.values, preds))


def calculate_shap_values(xgb_model: XGBClassifier,
                          X: pd.DataFrame
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate SHAP values for an XGBoost multiclass classifier using a shap TreeExplainer.

    Args:
        X: feature matrix [Samples X Features]

    Returns:
        shap_values: [Samples X Classes X Features]
        expected_logits: the expected output values of the classifier on the training set,
                         before softmaxing. [1 X Classes]
    """
    explainer = shap.TreeExplainer(xgb_model)
    # for some reason, if the explainer was never used, explainer.expected_value returns a single value
    # instead of an array of size [n_classes]. this is probably a bug in the shap package.
    # therefore, we "explain" a dummy sample before extracting the expected values.
    explainer.shap_values(np.ones((1, X.shape[1])))
    expected_logits = np.array(explainer.expected_value)[np.newaxis, :]

    shap_values = explainer.shap_values(X)
    shap_values = np.stack(shap_values, axis=1)
    return shap_values, expected_logits
