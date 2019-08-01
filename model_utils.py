import tensorflow as tf
import numpy as np
MAX_LOGITS = 70
MIN_LOGITS = -1

def weighted_MSE_loss(y_true, y_pred, weights):
    squared_error = (y_true - y_pred) ** 2
    weighted_squared_error = weights * squared_error
    weighted_MSE = tf.reduce_mean(weighted_squared_error, axis=[1, 2])
    weighted_MSE = tf.reshape(weighted_MSE, (-1,))
    return weighted_MSE


def zero_loss(y_true, y_pred):
    return tf.zeros(tf.shape(y_true)[0])

def shaps_to_probs(shaps, expected_logits):
    """
    shaps: shap values [Batch X Classes X Features]
    expected_logits: the (reshaped) result of explainer.expected_value [1 X Classes]
    returns probs: softmaxed probabilities [Batch X Classes]
    """
    logit_offsets = tf.reduce_sum(shaps, axis=2)
    logits = logit_offsets + expected_logits
    logits = tf.minimum(logits, MAX_LOGITS)
    logits = tf.maximum(logits, MIN_LOGITS)
    logits = logits - tf.reduce_min(logits, axis=1, keepdims=True)
    probs = tf.exp(logits)
    probs = probs / tf.reduce_sum(probs, axis=1, keepdims=True)
    return probs


def test_weighted_MSE_loss():
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


def test_shaps_to_probs(xgb_probs=None, shap_values=None, expected_logits=None):
    """

    :param xgb_probs:   xgb_probs = xgb.predict_proba(X)
    :param shap_values:
    :param expected_logits:
    :return:
    """
    sess = tf.Session()

    np_shaps = np.random.randn(2, 4, 6)
    np_expected_logits = np.random.randn(1, 4)

    t_shaps = tf.placeholder(tf.float32)
    t_expected_logits = tf.placeholder(tf.float32)
    t_res = shaps_to_probs(t_shaps, t_expected_logits)

    np_res = sess.run(t_res, feed_dict={t_shaps: np_shaps, t_expected_logits: np_expected_logits})
    print(np_res)

    if xgb_probs is not None and shap_values is not None and expected_logits is not None:
        shap_probs = sess.run(t_res, feed_dict={t_shaps: shap_values, t_expected_logits: expected_logits})
        print(np.allclose(shap_probs, xgb_probs))


if __name__ == '__main__':
    test_shaps_to_probs()
    test_weighted_MSE_loss()
