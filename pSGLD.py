import tensorflow as tf
from tensorflow_probability.python.math.diag_jacobian import diag_jacobian


class pSGLangevinDynamics(tf.keras.optimizers.Optimizer):
    """
    Preconditioned Stochastic Gradient Langevin Dynamics
    """
    def __init__(
        self,
        learning_rate=0.001,
        rho=0.9,
        epsilon=1e-7,
        data_size=1,
        burnin=0,
        parallel_iterations=10,
        include_gamma_term=False,
        jit_compile=True,
        name="pSGLD",
        **kwargs
    ):
        super().__init__(
            jit_compile=jit_compile,
            name=name,
            **kwargs
        )
        self.rho = rho # alpha parameter in preconditioner decay rate (alpha)
        self.epsilon = epsilon # diagonal bias (lambda)
        self.include_gamma_term = include_gamma_term
        self._data_size = tf.convert_to_tensor(data_size, name='data_size')
        self._burnin = tf.convert_to_tensor(burnin, name='burnin')
        self._learning_rate = self._build_learning_rate(learning_rate)
        self._parallel_iterations = parallel_iterations

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True

        self._velocities = []
        for var in var_list:
            self._velocities.append(
                self.add_variable_from_reference(var, "velocity")
            )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)

        stddev = tf.where(
            tf.squeeze(self.iterations > tf.cast(self._burnin, tf.int64)),
            tf.cast(tf.math.rsqrt(lr), gradient.dtype),
            tf.zeros([], gradient.dtype)
            )

        rho = self.rho

        var_key = self._var_key(variable)
        velocity = self._velocities[self._index_dict[var_key]]
                
        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            velocity.assign(rho * velocity)
            velocity.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * (1 - rho), gradient.indices
                )
            )
            denominator = velocity + self.epsilon
            denominator_slices = tf.gather(denominator, gradient.indices)
            preconditioner = tf.math.rsqrt(denominator_slices)
            if self.include_gamma_term:
                _, preconditioner_grads = diag_jacobian(xs=variable, ys=preconditioner, parallel_iterations=self._parallel_iterations)
                mean = 0.5 * (preconditioner * gradient * tf.cast(self._data_size, gradient.dtype) - preconditioner_grads[0])
            else:
                mean = 0.5 * (preconditioner * gradient * tf.cast(self._data_size, gradient.dtype))
            stddev *= tf.sqrt(preconditioner)
            result_shape = tf.broadcast_dynamic_shape(tf.shape(mean), tf.shape(stddev))
            noisy_grad = tf.random.normal(shape=result_shape, mean=mean, stddev=stddev, dtype=gradient.dtype)
            
            increment = tf.IndexedSlices(
                lr * noisy_grad,
                gradient.indices,
            )

            variable.scatter_add(-increment)
        else:
            # Dense gradients.
            velocity.assign(rho * velocity + (1 - rho) * tf.square(gradient))
            denominator = velocity + self.epsilon
            preconditioner = tf.math.rsqrt(denominator)
            if self.include_gamma_term:
                _, preconditioner_grads = diag_jacobian(xs=variable, ys=preconditioner, parallel_iterations=self._parallel_iterations)
                mean = 0.5 * (preconditioner * gradient * tf.cast(self._data_size, gradient.dtype) - preconditioner_grads[0])
            else:
                mean = 0.5 * (preconditioner * gradient * tf.cast(self._data_size, gradient.dtype))
            stddev *= tf.sqrt(preconditioner)
            result_shape = tf.broadcast_dynamic_shape(tf.shape(mean), tf.shape(stddev))
            noisy_grad = tf.random.normal(shape=result_shape, mean=mean, stddev=stddev, dtype=gradient.dtype)

            increment = lr * noisy_grad
            variable.assign_add(-increment)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "rho": self.rho,
                "epsilon": self.epsilon,
            }
        )
        return config
