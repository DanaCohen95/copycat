from models import get_student_nn_classifier, get_vanilla_nn_classifier
from data_utils import load_costa_rica_dataset, prepare_data
from xgboost_utils import fit_xgboost_classifier, calculate_shap_values, evaluate_xgboost_classifier, \
    save_xgboost_classifier, load_xgboost_classifier
import numpy as np
from sklearn.metrics import classification_report

use_weighted_shap_loss = False
num_features_to_use = 10;
model_type = "student"
assert model_type in ["student", "vanilla"]

X, y = load_costa_rica_dataset()
(n_samples, n_features, n_classes,
 X_train, X_valid, y_train, y_valid,
 y_train_onehot, y_valid_onehot, y_onehot,
 class_weights) = prepare_data(X, y)

if not use_weighted_shap_loss:
    class_weights = None

# xgb_model = load_xgboost_classifier("xgb_tomersh.xgb")
xgb_model = fit_xgboost_classifier(X_train, y_train)
save_xgboost_classifier(xgb_model, "xgb_tomersh.xgb")

shap_values_train, expected_logits = calculate_shap_values(xgb_model, X_train, num_features_to_use)
shap_values_valid, _ = calculate_shap_values(xgb_model, X_valid, num_features_to_use)
np.save('shap_values_train.npy', shap_values_train)
np.save('expected_logits.npy', expected_logits)
np.save('shap_values_valid.npy', shap_values_valid)
# shap_values_train = np.load('shap_values_train.npy')
# expected_logits = np.load('expected_logits.npy')
# shap_values_valid = np.load('shap_values_valid.npy')

# evaluate_xgboost_classifier(xgb_model, X_valid, y_valid)

if model_type == "student":
    model = get_student_nn_classifier(n_classes, n_features, num_features_to_use,
                                      expected_logits, class_weights=class_weights)

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
    print("=====")
    print("=====")
    print("=====")
    evaluate_xgboost_classifier(xgb_model, X_valid, y_valid)
