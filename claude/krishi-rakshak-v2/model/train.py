import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = "./data/plantvillage"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS_PHASE1 = 10
NUM_EPOCHS_PHASE2 = 10
MODEL_SAVE_PATH = "krishi_rakshak_v2.keras"
TFLITE_SAVE_PATH = "krishi_rakshak_v2.tflite"
CLASS_NAMES_PATH = "class_names.json"

# ── Data generators ───────────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2,
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

num_classes = len(train_gen.class_indices)
print(f"Classes found: {num_classes}")

# ── Model definition ──────────────────────────────────────────────────────────
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)
base_model.trainable = False  # Phase 1: freeze all base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# ── Phase 1: train head only ──────────────────────────────────────────────────
print("\n=== Phase 1: Training head ===")
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase1 = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy"),
]

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=NUM_EPOCHS_PHASE1,
    callbacks=callbacks_phase1,
)

# ── Phase 2: fine-tune last 30 layers ────────────────────────────────────────
print("\n=== Phase 2: Fine-tuning last 30 layers ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_phase2 = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor="val_accuracy"),
]

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=NUM_EPOCHS_PHASE2,
    callbacks=callbacks_phase2,
)

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n=== Classification Report ===")
val_gen.reset()
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_gen.classes

class_labels = list(train_gen.class_indices.keys())
print(classification_report(y_true, y_pred, target_names=class_labels))

# ── Export TFLite ─────────────────────────────────────────────────────────────
print("\n=== Exporting TFLite model ===")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(TFLITE_SAVE_PATH, "wb") as f:
    f.write(tflite_model)
print(f"TFLite model saved to {TFLITE_SAVE_PATH}")

# ── Save class names ──────────────────────────────────────────────────────────
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)
print(f"Class indices saved to {CLASS_NAMES_PATH}")
