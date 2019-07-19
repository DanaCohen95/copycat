from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report
from model_utils import shaps_to_probs, weighted_MSE_loss
from typing import Union
import pandas as pd


def get_vanilla_nn_classifier(n_classes: int,
                              n_features: int,
                              print_summary: bool = True,
                              ) -> keras.Sequential:
    """ Create a simple neural network for multiclass classification """
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=128, activation="relu", input_dim=n_features))
    model.add(keras.layers.Dense(units=n_classes, activation="softmax"))
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    if print_summary:
        model.summary()
    return model


def get_student_nn_classifier(n_classes: int,
                              n_features: int,
                              expected_logits: np.ndarray,
                              use_shap_loss: bool = True,
                              use_score_loss: bool = True,
                              class_weights: np.ndarray = None,
                              print_summary: bool = True,
                              ) -> keras.Model:
    """
    Create a shap-value-mimicking neural network for multiclass classification
    At least one of the use_****_loss arguments must be True.

    Args:
        expected_logits: the (reshaped) result of explainer.expected_value [1 X Classes]
        use_shap_loss: if True, train the net to mimic shap values using MSE loss.
        use_score_loss: if True, train the net for the explicit classification task using cross-entropy loss
        class_weights: weights for the MSE loss (shape [1 X Classes X 1]), or None for unweighted MSE loss

    Returns:
        A compiled NN model.
    """
    assert use_shap_loss or use_score_loss, "at least one of 'use_shap_loss', 'use_score_loss' must be True"

    l_input = keras.layers.Input(shape=(n_features,), name="input")
    l_hidden = keras.layers.Dense(units=128, activation="relu", name="hidden")(l_input)
    l_shaps_flat = keras.layers.Dense(units=n_classes * n_features, name="shaps_flat")(l_hidden)
    l_shaps = keras.layers.Reshape((n_classes, n_features), name="shaps")(l_shaps_flat)
    l_score = keras.layers.Lambda(
        lambda shaps: shaps_to_probs(shaps, expected_logits), output_shape=(n_classes,), name="score")(l_shaps)

    model = keras.models.Model(inputs=l_input, outputs=[l_score, l_shaps])

    if print_summary:
        model.summary()

    if class_weights is None:
        shap_loss = "mean_squared_error"
    else:
        def shap_loss(y_true, y_pred):
            return weighted_MSE_loss(y_true, y_pred, weights=class_weights)

    model.compile(optimizer="adam",
                  loss=["categorical_crossentropy", shap_loss],
                  loss_weights=[float(use_score_loss), float(use_shap_loss)],
                  metrics={"score": ["accuracy"]})
    return model


def evaluate_random_classifier(expected_logits: np.ndarray,
                               y_true: Union[np.ndarray, pd.Series],
                               n_clones: int = 1000
                               ) -> None:
    """
    Prints a classification_report for a random classifier, which randomly chooses labels
    for samples according to a non-uniform probability distribution.
    Can be used as a simple benchmark to see whether a student NN actually learned something
    beyond the already-known XGBoost expected class probabilities.

    Args:
        expected_logits: the expected output values of a classifier, before softmaxing.
                         can be derived from a SHAP explainer with explainer.expected_value
        y_true: class labels
        n_clones: the number of times every sample would be given a random label.
                  the higher n_clones, the estimation gets more accurate. 1000 is probably enough.
    """
    y_true = np.asarray(y_true)
    expected_logits = np.asarray(expected_logits).ravel()
    n_samples = len(y_true)
    n_classes = expected_logits.size
    base_probs = np.exp(expected_logits)
    base_probs /= base_probs.sum()
    random_preds = np.random.choice(np.arange(n_classes), p=base_probs, size=(n_samples, n_clones))
    targets = np.tile(y_true.reshape((-1, 1)), (1, n_clones))
    print(classification_report(np.ravel(targets), np.ravel(random_preds)))


def example_evaluate_random_classifier() -> None:
    """ compare a random classifier  """
    from data_utils import load_costa_rica_dataset, prepare_data
    from xgboost_utils import fit_xgboost_classifier, calculate_shap_values, evaluate_xgboost_classifier

    # load data, train xgboost model, calculate shap values
    X, y = load_costa_rica_dataset()
    (n_samples, n_features, n_classes,
     X_train, X_valid, y_train, y_valid,
     y_train_onehot, y_valid_onehot, y_onehot,
     class_weights) = prepare_data(X, y)

    xgb_model = fit_xgboost_classifier(X_train, y_train)
    shap_values, expected_logits = calculate_shap_values(xgb_model, X)

    # evaluate xgboost classifier
    print("\n", "evaluate xgboost classifier:")
    evaluate_xgboost_classifier(xgb_model, X_valid, y_valid)

    # evaluate random classifier
    print("\n", "evaluate random classifier (based of xgboost expected probabilities):")
    evaluate_random_classifier(expected_logits=expected_logits,
                               y_true=y_valid,
                               n_clones=1000)


if __name__ == '__main__':
    example_evaluate_random_classifier()
