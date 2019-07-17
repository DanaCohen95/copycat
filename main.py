from models import get_student_nn_classifier, get_vanilla_nn_classifier
from data_utils import load_costa_rica_dataset, prepare_data
from models import get_student_nn_classifier
from data_utils import prepare_data,over_sampling_by_big_class,load_dataset
from xgboost_utils import fit_xgboost_classifier, calculate_shap_values, evaluate_xgboost_classifier,save_xgboost_classifier
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

use_weighted_shap_loss = False
xgb_max_depth,xgb_n_estimators = 10,100
model_type = "vanilla"
assert model_type in ["student", "vanilla"]
dataset_name = 'otto'# 'costa_rica' 'safe_drive'

X, y = load_dataset(dataset_name)
(n_samples, n_features, n_classes,
 X_train, X_valid, y_train, y_valid,
 y_train_onehot, y_valid_onehot,
 class_weights) = prepare_data(X, y)

if not use_weighted_shap_loss:
    class_weights = None

xgb_model = fit_xgboost_classifier(X_train, y_train,max_depth=xgb_max_depth,n_estimators=xgb_n_estimators)
save_xgboost_classifier(xgb_model, f'xgb_depth_{xgb_max_depth}_estimators_{xgb_n_estimators}')
evaluate_xgboost_classifier(xgb_model, X_valid, y_valid)
shap_values_train, expected_logits = calculate_shap_values(xgb_model, X_train)
shap_values_valid, _ = calculate_shap_values(xgb_model, X_valid)


if model_type == "student":
    model = get_student_nn_classifier(n_classes, n_features, expected_logits, class_weights=class_weights)
    csv_logger = tf.keras.callbacks.CSVLogger('training.log')
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
