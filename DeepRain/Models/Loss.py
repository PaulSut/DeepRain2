import tensorflow as tf

def KLL(y_true, y_pred):
	kl = tf.keras.losses.KLDivergence(
    reduction=tf.keras.losses.Reduction.SUM)
	return kl(y_true, y_pred)

def NLL(y_true, y_hat):
    return -y_hat.prob(y_true)