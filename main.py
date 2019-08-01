from models import get_student_nn_classifier, get_vanilla_nn_classifier
from data_utils import prepare_data, load_dataset
from xgboost_utils import fit_xgboost_classifier, calculate_shap_values, \
    save_xgboost_classifier, load_xgboost_classifier
import numpy as np
from sklearn.metrics import classification_report, log_loss

load_saved_values = True
use_weighted_shap_loss = False
xgb_max_depth, xgb_n_estimators = 10, 100
NUM_EPOCHS = 50
num_shap_features = 10
model_type = "student"  # "student", "vanilla"
dataset_name = 'costa_rica'  # 'otto' 'costa_rica' 'safe_drive'
num_samples_to_keep = None  # None  1000

X, y = load_dataset(dataset_name)
n_samples, n_features, n_classes, \
X_train, X_valid, y_train, y_valid, \
y_train_onehot, y_valid_onehot, y_onehot, \
class_weights = prepare_data(X, y, num_samples_to_keep)

if not use_weighted_shap_loss:
    class_weights = None

if model_type == "student":
    if load_saved_values:
        xgb_model = load_xgboost_classifier(
            'experiments/{dataset_name}/xgb_depth_{xgb_max_depth}_estimators_{xgb_n_estimators}'.format(
                dataset_name=dataset_name, xgb_max_depth=xgb_max_depth, xgb_n_estimators=xgb_n_estimators))
        shap_values_train, expected_logits = calculate_shap_values(
            xgb_model, X_train, num_shap_features,
            file_path='experiments/{dataset_name}/train_shap_values.npy'.format(dataset_name=dataset_name))
        shap_values_valid, _ = calculate_shap_values(
            xgb_model, X_valid, num_shap_features,
            file_path='experiments/{dataset_name}/valid_shap_values.npy'.format(dataset_name=dataset_name))
    else:
        xgb_model = fit_xgboost_classifier(X_train, y_train, max_depth=xgb_max_depth, n_estimators=xgb_n_estimators)
        save_xgboost_classifier(xgb_model,
                                'experiments/{dataset_name}/xgb_depth_{xgb_max_depth}_estimators_{xgb_n_estimators}'.format(
                                    dataset_name=dataset_name, xgb_max_depth=xgb_max_depth,
                                    xgb_n_estimators=xgb_n_estimators))

        shap_values_train, expected_logits = calculate_shap_values(xgb_model, X_train, num_shap_features)
        shap_values_valid, _ = calculate_shap_values(xgb_model, X_valid, num_shap_features)
        np.save('experiments/{dataset_name}/train_shap_values.npy'.format(dataset_name=dataset_name), shap_values_train)
        np.save('experiments/{dataset_name}/expected_logits.npy'.format(dataset_name=dataset_name), expected_logits)
        np.save('experiments/{dataset_name}/valid_shap_values.npy'.format(dataset_name=dataset_name), shap_values_valid)

    model = get_student_nn_classifier(n_classes, n_features, num_shap_features,
                                      expected_logits, class_weights=class_weights)

    model.fit(X_train.values, [y_train_onehot, shap_values_train],
              validation_data=(X_valid.values, [y_valid_onehot, shap_values_valid]),
              epochs=NUM_EPOCHS)

    scores, shaps = model.predict(X_valid.values)
    preds = np.argmax(scores, axis=1)
    print("\n\n\n")
    print("Student NN classification report:")
    print(classification_report(y_valid.values, preds))
    print("log_loss:", log_loss(y_valid.values, scores))

    scores = xgb_model.predict_proba(X_valid)
    preds = np.argmax(scores, axis=1)
    print("\n\n\n")
    print("XGBoost classification report:")
    print(classification_report(y_valid.values, preds))
    print("log_loss:", log_loss(y_valid.values, scores))

elif model_type == "vanilla":
    model = get_vanilla_nn_classifier(n_classes, n_features)
    model.fit(X_train.values, y_train_onehot,
              validation_data=(X_valid.values, y_valid_onehot),
              epochs=NUM_EPOCHS)

    scores = model.predict(X_valid.values)
    preds = np.argmax(scores, axis=1)

    print("\n\n\n")
    print("Vanilla NN classification report:")
    print(classification_report(y_valid.values, preds))
    print("log_loss:", log_loss(y_valid.values, scores))
