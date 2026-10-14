import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="brain_tumor")
def to_255(image):
    """Nhân với 255 để đua về [0,255]"""
    return image * 255.0


def build_transfer_efficientnetb2(image_size=240, num_classes=3, weights="imagenet"):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.Lambda(to_255, name="to_255")(inputs)
    backbone = tf.keras.applications.EfficientNetB2(
        weights=weights, include_top=False, input_tensor=x
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="model4_original_efficientnetb2")

    model.compile(loss="categorical_crossentropy", optimizer="Adam", metrics=["accuracy"])
    return model
