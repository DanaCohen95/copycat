import warnings
# from xgboost import XGBClassifier
from multi_xgboost import MultiXGBClassifier
from multi_SHAP import MultiTreeExplainer
from sklearn.metrics import classification_report
import numpy as np
from multi_xgboost import MultiXGBClassifier
import pickle

def save_xgboost_classifier(xgb_model, name):
    pickle._dump(xgb_model, open(name+'.xgb', 'wb'))


def load_xgboost_classifier(name):
    xgb_model = pickle.load(open(name+'.xgb', 'rb'))
    return xgb_model


def fit_xgboost_classifier(X_train, y_train, max_depth=5,n_estimators=30 ):
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    xgb_model = MultiXGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=0.1, objective="multi:softmax")
    xgb_model.fit(X_train, y_train)
    print("finish fitting xgboost")
    return xgb_model


def evaluate_xgboost_classifier(xgb_model, X_valid, y_valid):
    preds = xgb_model.predict(X_valid)
    print(classification_report(y_valid.values, preds))


def calculate_shap_values(xgb_model, X):
    explainer = MultiTreeExplainer(xgb_model)
    explainer.shap_values(np.ones((1, X.shape[1])))
    expected_logits = np.array(explainer.expected_value)[np.newaxis, :]

    shap_values = explainer.shap_values(X)
    shap_values = np.stack(shap_values, axis=1)
    return shap_values, expected_logits
