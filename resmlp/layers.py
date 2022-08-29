"""base keras layers for resmlp."""
from typing import Optional, Type, Union

import tensorflow as tf
from einops.layers import tensorflow as tfeinsum
from tensorflow.keras import backend, constraints, initializers, layers, regularizers


class DiagonalAffine(layers.Layer):
    """Diagonal Affine transformation layer.

    Args:
        dims (int): [size of input feature]
        alpha_initializer (Union[str, Type[tf.keras.initializers.Initializer], initializers.Initializer], optional):
            Initializer for alpha vector. Defaults to "one".
        beta_initializer (Union[str, Type[tf.keras.initializers.Initializer], initializers.Initializer], optional):
            Initializer for beta vector. Defaults to "zero".
        use_beta (bool, optional): Boolean, whether the layer uses a beta vector.
            Defaults to True.
        alpha_regularizer (Union[str, Type[tf.keras.regularizers.Regularizer], tf.keras.regularizers.Regularizer], optional):
            Regularizer for alpha vector.Defaults to None.
        alpha_constraint (Union[str, Type[tf.keras.constraints.Constraint], tf.keras.constraints.Constraint], optional):
            Constraint for alpha vector. Defaults to None.
        beta_regularizer (Union[str, Type[tf.keras.regularizers.Regularizer], tf.keras.regularizers.Regularizer], optional):
            Regularizer for beta vector. Defaults to None.
        beta_constraint (Union[str, Type[tf.keras.constraints.Constraint], tf.keras.constraints.Constraint], optional):
            Constraint for beta vector. Defaults to None.

    Raises:
        ValueError: if dims is less than or equal to zero.
    """

    def __init__(
        self,
        dims: int,
        alpha_initializer: Union[str, Type[initializers.Initializer], initializers.Initializer] = "ones",
        beta_initializer: Union[str, Type[initializers.Initializer], initializers.Initializer] = "zero",
        use_beta: Optional[bool] = True,
        alpha_regularizer: Optional[Union[str, Type[regularizers.Regularizer], regularizers.Regularizer]] = None,
        alpha_constraint: Optional[Union[str, Type[constraints.Constraint], constraints.Constraint]] = None,
        beta_regularizer: Optional[Union[str, Type[regularizers.Regularizer], regularizers.Regularizer]] = None,
        beta_constraint: Optional[Union[str, Type[constraints.Constraint], constraints.Constraint]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dims = int(dims) if not isinstance(dims, int) else dims
        if self.dims <= 0:
            raise ValueError(f"Dimension must be greater than 0. Found {self.dims}")
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.alpha_initializer = initializers.get(alpha_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.use_beta = use_beta

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to should be defined. \
                Found `None`."
            )
        if last_dim != self.dims:
            raise ValueError(
                f"The last dimesion of the inputs should be equal to {self.dims}. \
                Found {last_dim}"
            )

        self.input_spec = layers.InputSpec(min_ndim=2, axes={-1: last_dim})
        self.alpha = self.add_weight(
            "alpha",
            shape=[self.dims],
            initializer=self.alpha_initializer,
            regularizer=self.alpha_regularizer,
            constraint=self.alpha_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_beta:
            self.beta = self.add_weight(
                "beta",
                shape=[self.dims],
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.beta = None
        self.built = True

    def call(self, inputs):  # pylint: disable=arguments-differ
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)
        affine = inputs * self.alpha
        if self.use_beta:
            affine = affine + self.beta
        return affine

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dims": self.dims,
                "use_beta": self.use_beta,
                "alpha_initializer": regularizers.serialize(self.alpha_initializer),
                "beta_initializer": regularizers.serialize(self.beta_initializer),
                "alpha_regularizer": regularizers.serialize(self.alpha_regularizer),
                "beta_regularizer": regularizers.serialize(self.beta_regularizer),
                "alpha_constraint": constraints.serialize(self.alpha_constraint),
                "beta_constraint": constraints.serialize(self.beta_constraint),
            }
        )
        return config


class PerSampleDropPath(layers.Layer):
    """
    Args:
        rate (float): Drop rate of layer Float between 0 and 1.
        seed (int, optional): A Python integer to use as random seed. Defaults to None.

    Raises:
        ValueError: if Rate is None or less than 0 or greater than or eqaul to one.
    """

    def __init__(self, rate: float, seed: int = None, **kwargs):
        super().__init__(**kwargs)
        if not rate and not isinstance(rate, float):
            raise ValueError("You must provide a rate")
        if not 0 <= rate < 1:
            raise ValueError("rate must be between 0 and 1")
        self.rate = rate
        self.seed = seed

    def call(self, inputs, training=None):  # pylint: disable=arguments-differ
        if training is None:
            training = backend.learning_phase()

        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        if training:
            batch_size = tf.shape(inputs)[0]
            survival_prob = 1 - self.rate
            random_tensor = survival_prob
            random_tensor = tf.cast(random_tensor, inputs.dtype)
            rank = inputs.shape.rank
            shape = (batch_size,) + (1,) * (rank - 1)
            random_tensor += tf.random.uniform(shape, dtype=inputs.dtype, seed=self.seed)
            binary_tensor = tf.floor(random_tensor)
            output = tf.math.divide(inputs, survival_prob) * binary_tensor
            return output

        return tf.identity(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate, "seed": self.seed})
        return config


def PatchEmbed(
    x: tf.Tensor,
    patch_width: int = 16,
    patch_height: int = 16,
    embed_dims: int = 768,
    flatten: Optional[bool] = True,
    name: str = "patch_embedding",
) -> tf.Tensor:
    """Patch Embedding layer.

    Args:
        x (tf.Tensor): Input tensor.
        patch_width (int, optional): Defaults to 16.
        patch_height (int, optional): Defaults to 16.
        embed_dims (int, optional): Defaults to 768.
        flatten (bool, optional): Defaults to True.
        name (str, optional): Defaults to patch_embedding.

    Returns:
        tf.Tensor: A tensor of rank 3+.
    """
    x = layers.Conv2D(
        filters=embed_dims,
        kernel_size=(patch_height, patch_width),
        strides=(patch_height, patch_width),
        name=name + "_conv_1",
    )(x)
    if flatten:
        x = tfeinsum.Rearrange("b h w c->b (h w) c")(x)
    return x
