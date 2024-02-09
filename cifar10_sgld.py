import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds

tf.config.set_visible_devices([], 'GPU')
# dont use eager execution
tf.compat.v1.disable_eager_execution()


# check if gpu is available
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# use cpu

ds_train, ds_test = tfds.load('cifar10', split=['train', 'test'], shuffle_files=True, as_supervised=True, batch_size=128)

n = 50000 # sample size
batch_size = 128
epochs = 10

# adding a prior to the weights
regularizer = tf.keras.regularizers.l2(0.01)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same', kernel_regularizer=regularizer, bias_regularizer=regularizer),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizer, bias_regularizer=regularizer),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizer, bias_regularizer=regularizer),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=regularizer, bias_regularizer=regularizer)
])


tf.compat.v1.enable_eager_execution

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=tf.constant(0.01),
    decay_steps=tf.constant(epochs*n//(batch_size)),
    decay_rate=tf.constant(0.9),
    staircase=False
)

class InverseTimeDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return self.initial_learning_rate / (1 + self.decay_rate * step / self.decay_steps)

lr_schedule = InverseTimeDecay(initial_learning_rate=0.01, decay_steps=epochs*n//batch_size, decay_rate=9.0)

lr_schedule(1000)

import matplotlib.pyplot as plt
import numpy as np



plt.plot(lr_schedule(tf.range(epochs*n//batch_size)))
plt.ylabel("Learning rate")
plt.xlabel("Step")
plt.show()

diagonal_bias = 1e-8 # lambda parameter in the paper (slightly different implementation)
preconditioner_decay_rate = 0.99 # alpha parameter in the paper
burnin = 100 # number of training steps before starting to add gradient noise
data_size = n

optimizer = tfp.optimizer.StochasticGradientLangevinDynamics(learning_rate=0.01/n, preconditioner_decay_rate=preconditioner_decay_rate, diagonal_bias=diagonal_bias, burnin=burnin, data_size=data_size)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)


# callback to record parameters
class RecordParametersCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.parameters = []
    def on_epoch_end(self, epoch, logs=None):
        self.parameters.append(model.get_weights())
    

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


callback = RecordParametersCallback()

model.fit(ds_train, epochs=10, validation_data=ds_test, callbacks=[callback], shuffle=True)

# get the parameters
    parameters = callback.parameters



