
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import json

# ── Config ───────────────────────────────────────────────
DATASET_DIR = "dataset/plantvillage dataset/color"
MODEL_SAVE  = "model/crop_model.h5"
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 15

os.makedirs("model", exist_ok=True)

# ── Data ─────────────────────────────────────────────────
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

NUM_CLASSES = len(train_gen.class_indices)
print(f"\n✅ Found {NUM_CLASSES} disease classes")
print(f"   Train : {train_gen.samples}")
print(f"   Val   : {val_gen.samples}\n")

with open("model/class_names.json", "w") as f:
    json.dump(train_gen.class_indices, f, indent=2)
print("✅ class_names.json saved\n")

# ── Model ─────────────────────────────────────────────────
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
print("✅ Model built\n")

# ── Callbacks ─────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(
        MODEL_SAVE,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )
]

# ── Train Phase 1 ONLY ────────────────────────────────────
print("=" * 50)
print("🚀 Training — Phase 1 only (safe, no fine-tuning)")
print("=" * 50 + "\n")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

best = max(history.history["val_accuracy"])
print(f"\n✅ Training complete!")
print(f"   Best val accuracy : {best:.2%}")
print(f"   Model saved to    : {MODEL_SAVE}")
print(f"\n🌾 Your model is ready — run whatsapp_bot.py now!")