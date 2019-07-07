from models import get_student_nn_classifier
from data_utils import load_costa_rica_dataset, prepare_data,load_sefe_drive_dataset
from xgboost_utils import fit_xgboost_classifier, calculate_shap_values
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report


# X, y = load_costa_rica_dataset()
X, y = load_sefe_drive_dataset()
(n_samples, n_features, n_classes,
 X_train, X_valid, y_train, y_valid,
 y_train_onehot, y_valid_onehot) = prepare_data(X, y)

xgb_model = fit_xgboost_classifier(X_train, y_train)
shap_values_train, expected_logits = calculate_shap_values(xgb_model, X_train)
shap_values_valid, _ = calculate_shap_values(xgb_model, X_valid)

model = get_student_nn_classifier(n_classes, n_features, expected_logits)
csv_logger = tf.keras.callbacks.CSVLogger('training.log')
model.fit(X_train.values, [y_train_onehot, shap_values_train],
          validation_data=(X_valid.values, [y_valid_onehot, shap_values_valid]),
          nb_epoch=50)

scores, shaps = model.predict(X_valid.values)
preds = np.argmax(scores, axis=1)
print(classification_report(y_valid.values, preds))
