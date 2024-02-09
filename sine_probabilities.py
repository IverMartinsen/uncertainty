import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt


def p(x, freq=1, decay_rate=1): 
    # probability of y=1 given x
    return np.exp(-decay_rate*x) * (np.sin(freq*x) + 1) / 2

n = 1000 # sample size

# data generation, x ~ Uniform(0, 10), y ~ N(p(x), 1)
x = tfp.distributions.Exponential(rate=1).sample(n)
y = tfp.distributions.Normal(loc=p(x, freq=1, decay_rate=0.2), scale=0.1).sample()

plt.plot(np.linspace(0, 10, 100), p(np.linspace(0, 10, 100), freq=1, decay_rate=0.2))
plt.scatter(x, y, alpha=0.1, s=10)
plt.legend([r'P(Y)', 'y'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# use neural network to approximate p(y|x)
width = 6
depth = 2

layers = [tf.keras.Input(shape=(1,))]
for i in range(depth):
    layers.append(tf.keras.layers.Dense(width, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
layers.append(tf.keras.layers.Dense(1, activation='linear'))

model = tf.keras.Sequential(layers)

# first fit the model using gradient descent
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

model.compile(optimizer=optimizer, loss='mse')

model.fit(x, y, epochs=100, batch_size=1000, verbose=1)

# plot the result
y_pred = model.predict(x)

plt.scatter(x, y, alpha=0.1, s=10)
plt.scatter(x, y_pred, alpha=0.1, s=10)
plt.plot(np.linspace(0, 10, 100), p(np.linspace(0, 10, 100), freq=1, decay_rate=0.2))
plt.legend(['y', r'$\hat{y}$'])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# now use HMC to sample from the posterior distribution of the weights
num_params = sum([np.prod(w.shape) for w in model.get_weights()])

shapes = [w.shape for w in model.get_weights()]

def f(x, w):
    _w = []
    idx = 0
    for s in shapes:
        _w.append(tf.reshape(w[idx:idx+np.prod(s)], s))
        idx += np.prod(s)
    
    
    
    model.set_weights(_w)
    return model.predict(x)


def joint_log_prob(w, x, y, num_params=num_params):
    prior = tfp.distributions.MultivariateNormalDiag(loc=np.zeros(num_params,), scale_diag=0.1*np.ones(num_params,))
    likelihood = tfp.distributions.Normal(loc=f(x, w), scale=0.1)
    # cast to float32 to avoid error
    #w = tf.cast(w, tf.float32)
    #y = tf.cast(y, tf.float32)
    loss1 = tf.cast(likelihood.log_prob(tf.reshape(y, [-1, 1])), tf.float64)
    loss2 = tf.cast(prior.log_prob(w), tf.float64)
    loss1 = tf.reduce_sum(loss1)
    #loss2 = tf.reduce_sum(loss2)
    #print(loss1.shape)
    #print(loss2.shape)
    return loss1 + loss2

def unnormalized_posterior(w):
    return joint_log_prob(w, x, y)

hmc_kernel = tfp.mcmc.NoUTurnSampler(
    target_log_prob_fn=unnormalized_posterior,
    step_size=0.01,
)

adaptive_hmc_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
    hmc_kernel,
    num_adaptation_steps=5,
    target_accept_prob=0.75,
)

#@tf.function
def run_chain(initial_state, num_results=2, num_burnin_steps=5):
  return tfp.mcmc.sample_chain(
    num_results=num_results,
    num_burnin_steps=num_burnin_steps,
    num_steps_between_results=0,
    current_state=initial_state,
    kernel=adaptive_hmc_kernel,
    trace_fn=lambda current_state, kernel_results: kernel_results)

# use current weights as initial state
initial_state = model.get_weights()
initial_state = [np.reshape(w, [-1]) for w in initial_state]
initial_state = [np.array(w) for w in initial_state]
initial_state = np.concatenate(initial_state, axis=0)
initial_state = tf.cast(initial_state, tf.float64)

initial_state = [np.random.normal(size=(num_params,))]
samples, kernel_results = run_chain(initial_state, num_results=10)


kernel_results.inner_results.is_accepted.numpy().mean()

for i in range(20):
    w = samples[i]
    y_pred = f(x, w)
    plt.scatter(x, y, alpha=0.1, s=10)
    plt.scatter(x, y_pred, alpha=0.1, s=10)
    plt.legend(['y', r'$\hat{y}$'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# glorot initializer for the first layer
fan_in = 1
fan_out = 3
limit = np.sqrt(6 / (fan_in + fan_out))
w = np.random.uniform(low=-limit, high=limit, size=(1, 3))
b = np.zeros((1, 3))

initial_state = []

for s in shapes:
    fan_in = s[0]
    fan_out = s[1]
    limit = np.sqrt(6 / (fan_in + fan_out))
    w = np.random.uniform(low=-limit, high=limit, size=s)
    b = np.zeros(s[1])
    initial_state.append(w)
    initial_state.append(b)

