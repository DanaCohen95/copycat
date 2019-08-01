from models import get_student_nn_classifier, get_vanilla_nn_classifier
from data_utils import load_costa_rica_dataset, prepare_data
from models import get_student_nn_classifier
from data_utils import prepare_data, load_dataset
from xgboost_utils import fit_xgboost_classifier, calculate_shap_values, \
    evaluate_xgboost_classifier, save_xgboost_classifier, load_xgboost_classifier
import numpy as np
from sklearn.metrics import classification_report

load_saved_values = True
use_weighted_shap_loss = False
xgb_max_depth, xgb_n_estimators = 10, 100
NUM_EPOCHS = 50
num_features_to_use = 10
model_type = "student"
assert model_type in ["student", "vanilla"]
dataset_name = 'costa_rica'

X, y = load_dataset(dataset_name)
n_samples, n_features, n_classes, \
X_train, X_valid, y_train, y_valid, \
y_train_onehot, y_valid_onehot, y_onehot, \
class_weights = prepare_data(X, y)

if not use_weighted_shap_loss:
    class_weights = None

if load_saved_values:
    xgb_model = load_xgboost_classifier(
        'experiments/{dataset_name}/xgb_depth_{xgb_max_depth}_estimators_{xgb_n_estimators}'.format(
            dataset_name=dataset_name, xgb_max_depth=xgb_max_depth, xgb_n_estimators=xgb_n_estimators))
    shap_values_train, expected_logits = calculate_shap_values(xgb_model, X_train, num_features_to_use,
                                                               file_path='experiments/{dataset_name}/train_shap_values.npy'.format(dataset_name=dataset_name))
    shap_values_valid, _ = calculate_shap_values(xgb_model, X_valid, num_features_to_use,
                                                 file_path='experiments/{dataset_name}/valid_shap_values.npy'.format(dataset_name=dataset_name))
else:
    xgb_model = fit_xgboost_classifier(X_train, y_train, max_depth=xgb_max_depth, n_estimators=xgb_n_estimators)
    save_xgboost_classifier(xgb_model,
        'experiments/{dataset_name}/xgb_depth_{xgb_max_depth}_estimators_{xgb_n_estimators}'.format(
            dataset_name=dataset_name, xgb_max_depth=xgb_max_depth, xgb_n_estimators=xgb_n_estimators))

    shap_values_train, expected_logits = calculate_shap_values(xgb_model, X_train, num_features_to_use)
    shap_values_valid, _ = calculate_shap_values(xgb_model, X_valid, num_features_to_use)
    np.save('experiments/{dataset_name}/train_shap_values.npy'.format(dataset_name=dataset_name), shap_values_train)
    np.save('experiments/{dataset_name}/expected_logits.npy'.format(dataset_name=dataset_name), expected_logits)
    np.save('experiments/{dataset_name}/shap_values_valid.npy'.format(dataset_name=dataset_name), shap_values_valid)

if model_type == "student":
    model = get_student_nn_classifier(n_classes, n_features, num_features_to_use,
                                      expected_logits, class_weights=class_weights)

    model.fit(X_train.values, [y_train_onehot, shap_values_train],
              validation_data=(X_valid.values, [y_valid_onehot, shap_values_valid]),
              epochs=NUM_EPOCHS)

    scores, shaps = model.predict(X_valid.values)
    preds = np.argmax(scores, axis=1)
    print("student nn classification report:")
    print(classification_report(y_valid.values, preds))
    print("xgboost classification report:")
    evaluate_xgboost_classifier(xgb_model, X_valid, y_valid)

elif model_type == "vanilla":
    model = get_vanilla_nn_classifier(n_classes, n_features)
    model.fit(X_train.values, y_train_onehot,
              validation_data=(X_valid.values, y_valid_onehot),
              epochs=NUM_EPOCHS)

    scores = model.predict(X_valid.values)
    preds = np.argmax(scores, axis=1)
    print("vanilla nn classification report:")
    print(classification_report(y_valid.values, preds))
    print("xgboost classification report:")
    evaluate_xgboost_classifier(xgb_model, X_valid, y_valid)
