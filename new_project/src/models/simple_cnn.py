import tensorflow as tf


def build_simple_cnn(image_size=224, learning_rate=1e-3):
    inputs = tf.keras.Input((image_size, image_size, 3))
    x = inputs

    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.Conv2D(
            filters, 3, padding="same", activation="relu"
        )(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="simple_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
