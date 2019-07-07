import warnings
from xgboost import XGBClassifier
import shap
from sklearn.metrics import classification_report
import numpy as np


def fit_xgboost_classifier(X_train, y_train):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    xgb_model = XGBClassifier(max_depth=5, n_estimators=30, learning_rate=0.1, objective="multi:softmax")
    xgb_model.fit(X_train, y_train)
    return xgb_model


def evaluate_xgboost_classifier(xgb_model, X_valid, y_valid):
    preds = xgb_model.predict(X_valid)
    print(classification_report(y_valid.values, preds))


def calculate_shap_values(xgb_model, X):
    explainer = shap.TreeExplainer(xgb_model)
    explainer.shap_values(np.ones((1, X.shape[1])))
    expected_logits = np.array(explainer.expected_value)[np.newaxis, :]

    shap_values = explainer.shap_values(X)
    shap_values = np.stack(shap_values, axis=1)
    return shap_values, expected_logits
