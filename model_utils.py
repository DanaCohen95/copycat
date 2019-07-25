import tensorflow as tf
import numpy as np


def weighted_MSE_loss(y_true: tf.Tensor,
                      y_pred: tf.Tensor,
                      weights: tf.Tensor
                      ) -> tf.Tensor:
    """
    Mean-Squared-Error Loss for 3-dim tensors, with weights on axis 1.

    Args:
        y_true: [Batch X d1 X d2]
        y_pred: [Batch X d1 X d2]
        weights: [1 X d1 X 1]

    Returns:
        weighted_MSE: loss per sample [Batch X 1]
    """
    squared_error = (y_true - y_pred) ** 2
    weighted_squared_error = weights * squared_error
    weighted_MSE = tf.reduce_mean(weighted_squared_error, axis=[1, 2])
    weighted_MSE = tf.reshape(weighted_MSE, (-1,))
    return weighted_MSE


def shaps_to_probs(shaps: tf.Tensor,
                   expected_logits: tf.Tensor
                   ) -> tf.Tensor:
    """
    Transform shap values to class probabilities.

    Args:
        shaps: shap values [Batch X Classes X Features]
        expected_logits: the (reshaped) result of explainer.expected_value [1 X Classes]

    Returns:
        probs: softmaxed probabilities [Batch X Classes]
    """
    MAX_LOGITS = 80.
    logit_offsets = tf.reduce_sum(shaps, axis=2)
    logits = logit_offsets + expected_logits
    logits = tf.minimum(logits, MAX_LOGITS)
    logits = logits - tf.reduce_min(logits, axis=1, keepdims=True)
    probs = tf.exp(logits)
    probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
    return probs


def test_weighted_MSE_loss_technical() -> None:
    """ technical test for the weighted_MSE_loss tensorflow function """
    sess = tf.Session()
    t_y_true = tf.placeholder(tf.float32)
    t_y_pred = tf.placeholder(tf.float32)
    t_weights = tf.placeholder(tf.float32)
    loss_func = lambda y_true, y_pred: weighted_MSE_loss(y_true, y_pred, t_weights)
    t_res = loss_func(t_y_true, t_y_pred)
    np_res = sess.run(t_res, feed_dict={t_y_true: np.random.rand(2, 4, 6),
                                        t_y_pred: np.random.rand(2, 4, 6),
                                        t_weights: np.random.rand(1, 4, 1)})
    print(np_res)
    print()


def test_shaps_to_probs_technical() -> None:
    """ technical test for the shaps_to_probs tensorflow function """
    sess = tf.Session()

    np_shaps = np.random.randn(2, 4, 6)
    np_expected_logits = np.random.randn(1, 4)

    t_shaps = tf.placeholder(tf.float32)
    t_expected_logits = tf.placeholder(tf.float32)
    t_res = shaps_to_probs(t_shaps, t_expected_logits)

    np_res = sess.run(t_res, feed_dict={t_shaps: np_shaps, t_expected_logits: np_expected_logits})
    print(np_res)
    print()


def test_shaps_to_probs_with_data() -> None:
    """
    test whether the shaps_to_probs tensorflow function actually calculates the correct
    class probabilities given actual shap values
    """
    from data_utils import load_costa_rica_dataset, prepare_data
    from xgboost_utils import fit_xgboost_classifier, calculate_shap_values

    # load data, train xgboost model, calculate shap values
    X, y = load_costa_rica_dataset()
    (n_samples, n_features, n_classes,
     X_train, X_valid, y_train, y_valid,
     y_train_onehot, y_valid_onehot, y_onehot,
     class_weights) = prepare_data(X, y)

    xgb_model = fit_xgboost_classifier(X_train, y_train)
    shap_values, expected_logits = calculate_shap_values(xgb_model, X)
    xgb_probs = xgb_model.predict_proba(X)

    # test shaps_to_probs
    sess = tf.Session()

    t_shaps = tf.placeholder(tf.float32)
    t_expected_logits = tf.placeholder(tf.float32)
    t_res = shaps_to_probs(t_shaps, t_expected_logits)

    shap_probs = sess.run(t_res, feed_dict={t_shaps: shap_values, t_expected_logits: expected_logits})
    print(np.allclose(shap_probs, xgb_probs))
    print()


if __name__ == '__main__':
    test_weighted_MSE_loss_technical()
    test_shaps_to_probs_technical()
    test_shaps_to_probs_with_data()
