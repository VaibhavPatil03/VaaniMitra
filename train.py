import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import pathlib
import matplotlib.pyplot as plt
import os

# Configuration
IMG_SIZE = (224, 224)  # MobileNetV2 input size
BATCH_SIZE = 32
EPOCHS = 20
MODEL_DIR = "Model"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data(data_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names

    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names

def create_transfer_model(input_shape, num_classes):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = True
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def save_class_names(class_names, file_path):
    with open(file_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

def main():
    data_dir = pathlib.Path("Data")
    train_ds, val_ds, class_names = load_data(data_dir)
    num_classes = len(class_names)

    save_class_names(class_names, os.path.join(MODEL_DIR, "labels.txt"))

    model = create_transfer_model((IMG_SIZE[0], IMG_SIZE[1], 3), num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "best_model.h5"), save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    model.save(os.path.join(MODEL_DIR, "keras_model.h5"))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.show()

if __name__ == "__main__":
    main()
