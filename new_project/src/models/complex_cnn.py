"""Selected BatchNorm Sequential-Parallel CNN; 456,755 parameters.

Configure the Keras backend and precision before calling build_complex_cnn.
The notebook contains the same model definition for standalone inspection.
"""

import keras
from keras import layers

def mbconv_block(x, name, bn_momentum=0.9, norm_epsilon=1e-3):
    """128 -> 256 -> 128 channels, depthwise3x3, linear projection, shortcut."""
    shortcut = x
    r = layers.Conv2D(256, 1, padding="same", use_bias=False,
                    name=f"{name}_expand")(x)
    r = layers.BatchNormalization(momentum=bn_momentum, epsilon=norm_epsilon, name=f"{name}_expand_bn")(r)
    r = layers.ReLU(max_value=6, name=f"{name}_expand_relu6")(r)
    r = layers.DepthwiseConv2D(3, padding="same", use_bias=False,
                            name=f"{name}_depthwise")(r)
    r = layers.BatchNormalization(momentum=bn_momentum, epsilon=norm_epsilon, name=f"{name}_depthwise_bn")(r)
    r = layers.ReLU(max_value=6, name=f"{name}_depthwise_relu6")(r)
    r = layers.Conv2D(128, 1, padding="same", use_bias=False,
                    name=f"{name}_project")(r)
    r = layers.BatchNormalization(momentum=bn_momentum, epsilon=norm_epsilon, name=f"{name}_project_bn")(r)
    return layers.Add(name=f"{name}_add")([shortcut, r])

def build_complex_cnn(decay_steps, learning_rate=2e-4, min_learning_rate=1e-6,
                    image_size=224, bn_momentum=0.9, dropout_rate=0.2,
                    accumulation_steps=4, norm_epsilon=1e-3):
    """Build and compile the two-branch CNN in one function."""
    inputs = keras.Input(shape=(image_size, image_size, 3), name="input_image")

    sequential = layers.Conv2D(32, 3, padding="same", use_bias=False, name="seq1_conv")(inputs)
    sequential = layers.BatchNormalization(momentum=bn_momentum, epsilon=norm_epsilon, name="seq1_bn")(sequential)
    sequential = layers.Activation("relu", name="seq1_relu")(sequential)

    sequential = layers.Conv2D(64, 3, padding="same", use_bias=False, name="seq2_conv")(sequential)
    sequential = layers.BatchNormalization(momentum=bn_momentum, epsilon=norm_epsilon, name="seq2_bn")(sequential)
    sequential = layers.Activation("relu", name="seq2_relu")(sequential)
    sequential = layers.MaxPooling2D(2, name="seq2_pool")(sequential)

    sequential = layers.Conv2D(128, 3, padding="same", use_bias=False, name="seq3_conv")(sequential)
    sequential = layers.BatchNormalization(momentum=bn_momentum, epsilon=norm_epsilon, name="seq3_bn")(sequential)
    sequential = layers.Activation("relu", name="seq3_relu")(sequential)
    sequential = layers.MaxPooling2D(2, name="seq3_pool")(sequential)
    # The parallel branch uses three different kernel sizes to capture multi-scale features.
    branches = []
    for kernel_size in (1, 3, 5):
        branch = layers.Conv2D(16, kernel_size, padding="same", use_bias=False,
                        name=f"par_{kernel_size}x{kernel_size}_conv")(inputs)
        branch = layers.BatchNormalization(momentum=bn_momentum, epsilon=norm_epsilon, name=f"par_{kernel_size}x{kernel_size}_bn")(branch)
        branch = layers.Activation("relu", name=f"par_{kernel_size}x{kernel_size}_relu")(branch)
        branches.append(branch)

    parallel = layers.Concatenate(name="par_concat")(branches)
    parallel = layers.MaxPooling2D(2, name="par_pool1")(parallel)
    parallel = layers.MaxPooling2D(2, name="par_pool")(parallel)

    x = layers.Concatenate(name="merge_concat")([sequential, parallel])
    x = layers.Conv2D(128, 3, padding="same", use_bias=False, name="fusion_conv")(x)
    x = layers.BatchNormalization(momentum=bn_momentum, epsilon=norm_epsilon, name="fusion_bn")(x)
    x = layers.Activation("relu", name="fusion_relu")(x)

    x = mbconv_block(x, name="mb1", bn_momentum=bn_momentum, norm_epsilon=norm_epsilon)
    x = mbconv_block(x, name="mb2", bn_momentum=bn_momentum, norm_epsilon=norm_epsilon)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense_128")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(3, activation="softmax", name="predictions", dtype="float32")(x)
    model = keras.Model(inputs, outputs, name="complex_sequential_parallel_batchnorm_cnn")

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        alpha=min_learning_rate / learning_rate,
    )
    model.compile(optimizer=keras.optimizers.Adam(
                    learning_rate=lr_schedule,
                    gradient_accumulation_steps=accumulation_steps,
                ),
                loss="categorical_crossentropy", metrics=["accuracy"])
    return model
