from models import get_student_nn_classifier, get_vanilla_nn_classifier
from data_utils import load_costa_rica_dataset, prepare_data
from xgboost_utils import fit_xgboost_classifier, calculate_shap_values
import numpy as np
from sklearn.metrics import classification_report

model_type = "student"
assert model_type in ["student", "vanilla"]

X, y = load_costa_rica_dataset()
(n_samples, n_features, n_classes,
 X_train, X_valid, y_train, y_valid,
 y_train_onehot, y_valid_onehot) = prepare_data(X, y)

xgb_model = fit_xgboost_classifier(X_train, y_train)
shap_values_train, expected_logits = calculate_shap_values(xgb_model, X_train)
shap_values_valid, _ = calculate_shap_values(xgb_model, X_valid)

if model_type == "student":
    model = get_student_nn_classifier(n_classes, n_features, expected_logits)

    model.fit(X_train.values, [y_train_onehot, shap_values_train],
              validation_data=(X_valid.values, [y_valid_onehot, shap_values_valid]),
              epochs=50)

    scores, shaps = model.predict(X_valid.values)
    preds = np.argmax(scores, axis=1)
    print(classification_report(y_valid.values, preds))

elif model_type == "vanilla":
    model = get_vanilla_nn_classifier(n_classes, n_features)
    model.fit(X_train.values, y_train_onehot,
              validation_data=(X_valid.values, y_valid_onehot),
              epochs=50)

    scores = model.predict(X_valid.values)
    preds = np.argmax(scores, axis=1)
    print(classification_report(y_valid.values, preds))
