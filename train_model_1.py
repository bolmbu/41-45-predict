from pathlib import Path
import json
import random
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------------------
# CONFIG
# ------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
LABELS_PATH = BASE_DIR / "labels.json"

IMG_SIZE = 28
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 32

# ------------------------------------
# Load Labels
# ------------------------------------
def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------------------------
# Preprocess each image
# ------------------------------------
def preprocess_file(path: Path):
    img = Image.open(path).convert("L")
    img = ImageOps.invert(img)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(IMG_SIZE, IMG_SIZE, 1)

# ------------------------------------
# Load whole dataset
# ------------------------------------
def load_dataset(labels):
    X, y = [], []

    for idx, label in enumerate(labels):
        label_dir = DATASET_DIR / label
        files = list(label_dir.glob("*.png")) + list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.jpeg"))
        print(f"{label}: {len(files)} images")

        for f in files:
            try:
                X.append(preprocess_file(f))
                y.append(idx)
            except Exception as e:
                print(f"Skip {f}: {e}")

    if len(X) == 0:
        raise RuntimeError("ยังไม่มี dataset — เข้าหน้า /collect ก่อน")

    return np.array(X), np.array(y)


# ------------------------------------
# Squeeze-and-Excitation block (SE) ★ เพิ่มความแม่นยำ
# ------------------------------------
def se_block(x, reduction=16):
    filters = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // reduction, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    return layers.Multiply()([x, se])


# ------------------------------------
# Stronger CNN model with BatchNorm + SE Attention
# ------------------------------------
def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = se_block(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = se_block(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.45)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0007, weight_decay=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ------------------------------------
# Save Training Graphs
# ------------------------------------
def save_training_plot(history):
    REPORT_DIR.mkdir(exist_ok=True)

    # Accuracy Plot
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "accuracy_plot.png", dpi=160)
    plt.close()

    # Loss Plot
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "loss_plot.png", dpi=160)
    plt.close()


# ------------------------------------
# MAIN TRAINING
# ------------------------------------
def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    labels = load_labels()
    X, y = load_dataset(labels)

    # Dataset stats
    counts = {label: int(np.sum(y == i)) for i, label in enumerate(labels)}
    print("\nDataset counts:", counts)

    # Split
    stratify = y if min(counts.values()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify
    )

    # Stronger Augmentation ★
    datagen = ImageDataGenerator(
        rotation_range=12,
        width_shift_range=0.12,
        height_shift_range=0.12,
        zoom_range=0.15,
        shear_range=8,
        brightness_range=[0.7, 1.3]
    )
    datagen.fit(X_train)

    train_flow = datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

    # Build model
    model = build_model(num_classes=len(labels))

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.6,
            patience=5,
            min_lr=1e-6
        )
    ]

    # Train
    history = model.fit(
        train_flow,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")

    # Save model
    model_path = MODEL_DIR / "thai_digit_model.keras"
    model.save(model_path)
    print(f"Saved model to: {model_path}")

    # Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Report
    report = classification_report(y_test, y_pred, target_names=labels, digits=4)
    (REPORT_DIR / "classification_report.txt").write_text(
        f"Accuracy: {test_acc:.4f}\n\n{report}",
        encoding="utf-8"
    )
    print("\n" + report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    save_training_plot(history)
    print("\nReports saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()
