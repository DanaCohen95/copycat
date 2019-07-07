from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report
from model_utils import shaps_to_probs


def get_vanilla_nn_classifier(n_classes, n_features):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=128, activation="relu", input_dim=n_features))
    model.add(keras.layers.Dense(units=n_classes, activation="softmax"))
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model


def get_student_nn_classifier(n_classes, n_features, expected_logits,
                              use_shap_loss=True, use_target_loss=True):
    assert use_shap_loss or use_target_loss, "at least one of 'use_shap_loss', 'use_target_loss' must be True"

    l_input = keras.layers.Input(shape=(n_features,), name="input")
    l_hidden = keras.layers.Dense(units=128, activation="relu", name="hidden")(l_input)
    l_shaps_flat = keras.layers.Dense(units=n_classes * n_features, name="shaps_flat")(l_hidden)
    l_shaps = keras.layers.Reshape((n_classes, n_features), name="shaps")(l_shaps_flat)
    l_score = keras.layers.Lambda(
        lambda shaps: shaps_to_probs(shaps, expected_logits), output_shape=(n_classes,), name="score")(l_shaps)

    model = keras.models.Model(inputs=l_input, outputs=[l_score, l_shaps])
    model.summary()
    model.compile(optimizer="adam",
                  loss=["categorical_crossentropy", "mean_squared_error"],
                  loss_weights=[float(use_target_loss), float(use_shap_loss)],
                  metrics={"score": ["accuracy"]})
    return model


def evaluate_random_classifier(expected_logits, y_true, n_clones=1000):
    """
    :param expected_logits:  explainer.expected_value
    :param y_true:
    :param n_clones:
    :return:
    """
    y_true = np.asarray(y_true)
    n_samples = len(y_true)
    n_classes = expected_logits.size
    base_probs = np.exp(expected_logits).ravel()
    base_probs /= base_probs.sum()
    random_preds = np.random.choice(np.arange(n_classes), p=base_probs, size=(n_samples, n_clones))
    targets = np.tile(y_true.reshape((-1, 1)), (1, n_clones))
    print(classification_report(np.ravel(targets), np.ravel(random_preds)))
