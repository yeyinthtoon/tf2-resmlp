import tensorflow as tf
from resmlp.layers import DiagonalAffine, PerSampleDropPath
from tensorflow import test
from tensorflow.keras import initializers, layers

delattr(test.TestCase, "test_session")


class TestDiagonalAffine(test.TestCase):
    def test_is_same_result_as_dense_layer(self):
        dims = 10
        for i in range(3):
            tensor_data = tf.random.uniform(shape=[5] + [i for _ in range(i)] + [dims])
            fake_initializer = tf.linalg.diag(tf.ones(dims))
            result = DiagonalAffine(dims=dims)(tensor_data)
            target = layers.Dense(
                units=dims,
                kernel_initializer=initializers.Constant(fake_initializer),
            )(tensor_data)
            self.assertAllEqual(
                result,
                target,
                f"Test fail at tensor rank {tf.rank(tensor_data)}",
            )

    def test_is_same_result_as_dense_layer_autograph(self):
        dims = 10
        tensor_data = tf.random.uniform(shape=[5, dims])
        fake_initializer = tf.linalg.diag(tf.ones(dims))

        diagonal_affine = DiagonalAffine(dims=dims)
        diagonal_affine.build([dims])
        dense = layers.Dense(
            units=dims,
            kernel_initializer=initializers.Constant(fake_initializer),
        )
        dense.build([dims])

        @tf.function
        def diagonal_affine_autograph(tensor_data):
            return diagonal_affine(tensor_data)

        @tf.function
        def dense_autograph(tensor_data):
            return dense(tensor_data)

        result = diagonal_affine_autograph(tensor_data)
        target = dense_autograph(tensor_data)
        self.assertAllEqual(result, target)

    def test_load_from_config(self):
        dims = 10
        tensor_data = tf.random.uniform(shape=[5, 10])
        fake_initializer = tf.fill(dims, 1e-3)
        diagonal_affine = DiagonalAffine(
            dims=dims,
            alpha_initializer=initializers.Constant(fake_initializer),
        )
        target = diagonal_affine(tensor_data)
        result = DiagonalAffine.from_config(diagonal_affine.get_config())(tensor_data)
        self.assertAllEqual(result, target)

    def test_beta_not_use(self):
        dims = 10
        diagonal_affine = DiagonalAffine(dims=dims, use_beta=False)
        diagonal_affine.build([dims])
        self.assertAllEqual(dims, diagonal_affine.count_params())

    def test_beta_use(self):
        dims = 10
        diagonal_affine = DiagonalAffine(dims=dims, use_beta=True)
        diagonal_affine.build([dims])
        self.assertAllEqual(2 * dims, diagonal_affine.count_params())

    def test_raise_type_error_if_dims_None(self):
        with self.assertRaises(TypeError):
            DiagonalAffine(dims=None)

    def test_raise_value_error_if_dims_zero(self):
        with self.assertRaises(ValueError):
            DiagonalAffine(dims=0)

    def test_raise_value_error_if_inputs_has_different_dims(self):
        dims = 10
        with self.assertRaises(ValueError):
            tensor_data = tf.random.uniform(shape=[5, dims])
            _ = DiagonalAffine(dims=dims + 1)(tensor_data)


class TestPerSampleDropPath(test.TestCase):
    def test_is_output_same_as_input_at_test_time(self):
        for i in range(3):
            tensor_data = tf.random.uniform(shape=[5] + [i for _ in range(i)] + [10])
            rate = 0.2
            droppath = PerSampleDropPath(rate=rate)
            result = droppath(tensor_data, training=False)
            self.assertAllEqual(
                result,
                tensor_data,
                f"Test fail at tensor rank {tf.rank(tensor_data)}",
            )

    def test_is_output_different_from_input_at_train_time(self):
        for i in range(3):
            tensor_data = tf.random.uniform(shape=[5] + [i for _ in range(i)] + [10])
            rate = 0.2
            droppath = PerSampleDropPath(rate=rate)
            result = droppath(tensor_data, training=True)
            self.assertNotAllEqual(
                result,
                tensor_data,
                f"Test fail at tensor rank {tf.rank(tensor_data)}",
            )

    def test_is_output_same_as_input_at_zero_drop_rate(self):
        for i in range(3):
            tensor_data = tf.random.uniform(shape=[5] + [i for _ in range(i)] + [10])
            rate = 0.0
            droppath = PerSampleDropPath(rate=rate)
            result = droppath(tensor_data, training=True)
            self.assertAllEqual(
                result,
                tensor_data,
                f"Test fail at tensor rank {tf.rank(tensor_data)}",
            )

    def test_is_output_same_as_input_at_test_time_autograph(self):
        rate = 0.2
        droppath = PerSampleDropPath(rate=rate)

        @tf.function
        def droppath_autograph(tensordata, training=False):
            return droppath(tensor_data, training=training)

        for i in range(3):
            tensor_data = tf.random.uniform(shape=[5] + [i for _ in range(i)] + [10])
            result = droppath_autograph(tensor_data, training=False)
            self.assertAllEqual(
                result,
                tensor_data,
                f"Test fail at tensor rank {tf.rank(tensor_data)}",
            )

    def test_is_output_different_from_input_at_train_time_autograph(
        self,
    ):
        rate = 0.2
        droppath = PerSampleDropPath(rate=rate)

        @tf.function
        def droppath_autograph(tensordata, training=False):
            return droppath(tensor_data, training=training)

        for i in range(3):
            tensor_data = tf.random.uniform(shape=[5] + [i for _ in range(i)] + [10])
            result = droppath_autograph(tensor_data, training=True)
            self.assertNotAllEqual(
                result,
                tensor_data,
                f"Test fail at tensor rank {tf.rank(tensor_data)}",
            )

    def test_is_output_same_as_input_at_zero_drop_rate_autograph(self):
        rate = 0.0
        droppath = PerSampleDropPath(rate=rate)

        @tf.function
        def droppath_autograph(tensordata, training=False):
            return droppath(tensor_data, training=training)

        for i in range(3):
            tensor_data = tf.random.uniform(shape=[5] + [i for _ in range(i)] + [10])
            result = droppath_autograph(tensor_data, training=True)
            self.assertAllEqual(
                result,
                tensor_data,
                f"Test fail at tensor rank {tf.rank(tensor_data)}",
            )

    def test_raise_value_error_if_rate_is_none(self):
        with self.assertRaises(ValueError):
            PerSampleDropPath(rate=None)

    def test_raise_value_error_if_rate_is_less_than_0(self):
        with self.assertRaises(ValueError):
            PerSampleDropPath(rate=-1)

    def test_raise_value_error_if_rate_is_equal_to_1(self):
        with self.assertRaises(ValueError):
            PerSampleDropPath(rate=1)

    def test_raise_value_error_if_rate_is_greater_than_1(self):
        with self.assertRaises(ValueError):
            PerSampleDropPath(rate=1.1)
