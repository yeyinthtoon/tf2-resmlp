import tensorflow as tf
from einops.layers import tensorflow as tfeinsum
from tensorflow.keras import initializers, layers

from .layers import DiagonalAffine, PatchEmbed, PerSampleDropPath


def mlp_block(x, hidden_units, out_units, activation, dropout_rate, name=None):
    x = layers.Dense(units=hidden_units, name=name + "_dense_1")(x)
    x = layers.Activation(activation, name=name + "_act_1")(x)
    x = layers.Dropout(dropout_rate, name=name + "_dropout_1")(x)
    x = layers.Dense(units=out_units, name=name + "_dense_2")(x)
    x = layers.Dropout(dropout_rate, name=name + "_dropout_2")(x)
    return x


def layers_scale_mlp_blocks(
    x,
    dims,
    dropout_rate,
    drop_path_rate,
    activation,
    init_values,
    num_patches,
    name=None,
):
    inputs = tf.identity(x)
    x = DiagonalAffine(dims, name=name + "_affine_1")(x)
    x = tf.transpose(x, (0, 2, 1), name=name + "_transpose_1")
    x = layers.Dense(num_patches, name=name + "_dense_1")(x)
    x = tf.transpose(x, (0, 2, 1), name=name + "_transpose_2")
    x = DiagonalAffine(
        dims,
        alpha_initializer=initializers.Constant(tf.fill([dims], init_values)),
        use_beta=False,
        name=name + "_affine_2",
    )(x)
    x = PerSampleDropPath(drop_path_rate, name=name + "_drop_path_1")(x)
    x = layers.Add(name=name + "_add_1")([inputs, x])
    z = x
    x = DiagonalAffine(dims, name=name + "_affine_3")(x)
    x = mlp_block(x, 4 * dims, dims, activation, dropout_rate, name=name + "_mlp")
    x = DiagonalAffine(
        dims,
        alpha_initializer=initializers.Constant(tf.fill([dims], init_values)),
        use_beta=False,
        name=name + "_affine_4",
    )(x)
    x = PerSampleDropPath(drop_path_rate, name=name + "_drop_path_2")(x)
    x = layers.Add(name=name + "_add_2")([z, x])
    return x


def resmlp(
    input_shape=(224, 224, 3),
    patch_width=16,
    patch_height=16,
    num_classes=1000,
    embed_dims=768,
    depth=12,
    dropout_rate=0.0,
    drop_path_rate=0.0,
    init_scale=1e-4,
    activation="gelu",
    include_top=True,
    model_name=None,
):
    inputs = x = tf.keras.layers.Input(shape=input_shape)
    x = PatchEmbed(
        x,
        patch_width=patch_width,
        patch_height=patch_height,
        embed_dims=embed_dims,
        name="patch_embedding",
    )
    shape = x.get_shape()
    for i in range(depth):
        x = layers_scale_mlp_blocks(
            x,
            dims=embed_dims,
            dropout_rate=dropout_rate,
            drop_path_rate=drop_path_rate,
            init_values=init_scale,
            activation=activation,
            num_patches=shape[1],
            name=f"block_{i}",
        )

    x = DiagonalAffine(dims=embed_dims, name="feature_affine")(x)
    if include_top:
        x = tfeinsum.Reduce("b n c -> b c", "mean")(x)
        x = layers.Dense(num_classes, name="predictions")(x)

    model = tf.keras.models.Model(inputs, x, name=model_name)
    return model


def ResMlp12(
    input_shape=(224, 224, 3),
    patch_width=16,
    patch_height=16,
    embed_dims=384,
    model_name="resmlp12",
    **kwargs,
):
    return resmlp(
        input_shape=input_shape,
        patch_width=patch_width,
        patch_height=patch_height,
        embed_dims=embed_dims,
        depth=12,
        init_scale=1e-1,
        model_name=model_name,
        **kwargs,
    )


def ResMlp24(
    input_shape=(224, 224, 3),
    patch_width=16,
    patch_height=16,
    embed_dims=384,
    model_name="resmlp24",
    **kwargs,
):
    return resmlp(
        input_shape=input_shape,
        patch_width=patch_width,
        patch_height=patch_height,
        embed_dims=embed_dims,
        depth=24,
        init_scale=1e-5,
        model_name=model_name,
        **kwargs,
    )


def ResMlp36(
    input_shape=(224, 224, 3),
    patch_width=16,
    patch_height=16,
    embed_dims=384,
    model_name="resmlp36",
    **kwargs,
):
    return resmlp(
        input_shape=input_shape,
        patch_width=patch_width,
        patch_height=patch_height,
        embed_dims=embed_dims,
        depth=36,
        init_scale=1e-6,
        model_name=model_name,
        **kwargs,
    )


def ResMlpB24(
    input_shape=(224, 224, 3),
    patch_width=8,
    patch_height=8,
    embed_dims=768,
    model_name="resmlpB24",
    **kwargs,
):
    return resmlp(
        input_shape=input_shape,
        patch_width=patch_width,
        patch_height=patch_height,
        embed_dims=embed_dims,
        depth=24,
        init_scale=1e-6,
        model_name=model_name,
        **kwargs,
    )
