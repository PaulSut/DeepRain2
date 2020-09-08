import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def ZeroInflated_Poisson():
	return tfp.layers.DistributionLambda(
        name="DistributionLayer",
        make_distribution_fn=lambda t: tfp.distributions.Independent(
        tfd.Mixture(
            cat=tfd.Categorical(probs=tf.stack([1-t[...,0:1], t[...,0:1]],axis=-1)),
            components=[tfd.Deterministic(loc=tf.zeros_like(t[...,0:1])),
            tfd.Poisson(rate=tf.math.softplus(t[...,1:2]))]),
        name="ZeroInflated",reinterpreted_batch_ndims=0 ))



def ZeroInflated_Binomial():
    return tfp.layers.DistributionLambda(
        name="DistributionLayer",
        make_distribution_fn=lambda t: tfp.distributions.Independent(
        tfd.Mixture(
                    cat=tfd.Categorical(tf.stack([1-t[...,0:1],
                        t[...,0:1]],axis=-1)),
                    components=[tfd.Deterministic(loc=tf.zeros_like(t[...,0:1])),
                    tfp.distributions.NegativeBinomial(
                    total_count=t[..., 1:2], \
                    logits=t[..., 2:])]),
        #name="ZeroInflated_Binomial",reinterpreted_batch_ndims=-1 ))
        name="ZeroInflated_Binomial",reinterpreted_batch_ndims=0 ))