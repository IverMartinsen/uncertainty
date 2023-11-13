import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

# data generation
n = 1000 # sample size
d = 100 # number of predictors
k = 4 # number of classes

# hyperparameters
prior_scale = 10
num_burnin_steps = 10
target_accept_prob = 0.75
num_results = 10
thinning = 0

if __name__ == '__main__':

    print("Generating data...")
    print("Number of predictors:", d)
    print("Number of classes:", k)
    print("Sample size:", n)

    x = tfp.distributions.Uniform(low=-1, high=1).sample((n, d)) # sample some random predictors
    w = tfp.distributions.Normal(loc=0, scale=1).sample((d, k)) # sample some random weights
    y = tfp.distributions.Categorical(logits=tf.matmul(x, w)).sample() # sample some random responses

    print("Class proportions:")
    for i in range(k):
        print('Class {}: {}'.format(i, np.mean(y == i)))


    def joint_log_prob(w, x, y):
        """
        Joint log probability optimization function, i.e. the log of the unnormalized posterior.
        """
        # diagonal normal prior for all weights
        prior = tfp.distributions.MultivariateNormalDiag(loc=np.zeros((k*d, ), dtype=np.float32), scale_diag=prior_scale*np.ones((k*d, ), dtype=np.float32))
        # reshape to a matrix with vectors of length d for each class
        w = tf.reshape(w, (d, k))
        # compute the categorical likelihood
        likelihood = tfp.distributions.Categorical(logits=tf.matmul(x, w))
        # reshape back to a parameter vector
        w = tf.reshape(w, (-1, 1))
        # compute the log posterior
        loss = tf.reduce_sum(likelihood.log_prob(y)) + tf.reduce_sum(prior.log_prob(w))
        return loss


    def unnormalized_posterior(w):
        """Wrapper function for the joint log probability function."""
        return joint_log_prob(w, x, y)


    adaptive_hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        tfp.mcmc.NoUTurnSampler(target_log_prob_fn=unnormalized_posterior, step_size=0.01),
        num_adaptation_steps=int(num_burnin_steps * 0.8),
        target_accept_prob=target_accept_prob,
    )


    @tf.function
    def run_chain(initial_state, num_results, num_burnin_steps, thinning):
        """Run the chain with the given number of steps and burnin steps."""
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            num_steps_between_results=thinning,
            current_state=initial_state,
            kernel=adaptive_hmc_kernel,
            trace_fn=lambda _, kernel_results: kernel_results
            )

    initial_state = tfp.distributions.Normal(loc=0, scale=1).sample((k*d, ))
    print("Starting MCMC...")
    samples, kernel_results = run_chain(initial_state, num_results=num_results, num_burnin_steps=num_burnin_steps, thinning=thinning)
    print("MCMC finished.")

    print("Acceptance rate:", kernel_results.inner_results.is_accepted.numpy().mean())

    # compute accuracy and weight statistics
    samples = samples.numpy().reshape((-1, d, k))
    w_mean = samples.mean(axis=0)
    w_var = samples.var(axis=0)
    w_t = w_mean / (w_var / np.sqrt(num_results))

    y_pred = tf.matmul(x, w_mean).numpy().argmax(axis=1)
    acc = np.mean(y_pred == tf.cast(y, tf.float32))
    print("Accuracy:", acc)

    # plot weight statistics
    fig, ax = plt.subplots(1, 3, figsize=(10, 10))

    im = ax[0].imshow(w_mean, cmap='RdBu_r')
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Posterior mean")
    ax[0].axis('off')

    im = ax[1].imshow(w_var)
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Posterior variance")
    ax[1].axis('off')

    im = ax[2].imshow(w_t, cmap='RdBu_r')
    fig.colorbar(im, ax=ax[2])
    ax[2].set_title("T-statistic")
    ax[2].axis('off')

    plt.show()
