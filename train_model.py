from pathlib import Path
import json
import random

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
LABELS_PATH = BASE_DIR / "labels.json"

IMG_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 60
BATCH_SIZE = 16


def load_labels():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def center_crop_handwriting(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.invert(img)

    arr = np.array(img)
    mask = arr > 30

    if not mask.any():
        return img.resize((IMG_SIZE, IMG_SIZE))

    ys, xs = np.where(mask)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    margin = 12
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(arr.shape[1] - 1, x2 + margin)
    y2 = min(arr.shape[0] - 1, y2 + margin)

    crop = img.crop((x1, y1, x2 + 1, y2 + 1))

    w, h = crop.size
    side = max(w, h)
    square = Image.new("L", (side, side), 0)
    square.paste(crop, ((side - w) // 2, (side - h) // 2))

    pad = max(4, side // 12)
    padded = Image.new("L", (side + pad * 2, side + pad * 2), 0)
    padded.paste(square, (pad, pad))

    return padded.resize((IMG_SIZE, IMG_SIZE))


def preprocess_file(path: Path):
    img = Image.open(path).convert("L")
    img = center_crop_handwriting(img)
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(IMG_SIZE, IMG_SIZE, 1)


def load_dataset(labels):
    X, y = [], []

    for idx, label in enumerate(labels):
        label_dir = DATASET_DIR / label
        files = (
            list(label_dir.glob("*.png"))
            + list(label_dir.glob("*.jpg"))
            + list(label_dir.glob("*.jpeg"))
        )

        print(f"{label}: {len(files)} images")

        for file in files:
            try:
                X.append(preprocess_file(file))
                y.append(idx)
            except Exception as e:
                print(f"Skip {file}: {e}")

    if len(X) == 0:
        raise RuntimeError("ยังไม่มี dataset ให้เข้า Admin Dashboard เพื่อบันทึกรูปก่อน")

    return np.array(X), np.array(y)


def build_model(num_classes):
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.08),
    ], name="augmentation")

    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        data_augmentation,

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.20),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        layers.Flatten(),
        layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=regularizers.l2(0.001)
        ),
        layers.Dropout(0.40),
        layers.Dense(num_classes, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def save_training_plot(history):
    REPORT_DIR.mkdir(exist_ok=True)

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


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    labels = load_labels()
    X, y = load_dataset(labels)

    counts = {label: int(np.sum(y == i)) for i, label in enumerate(labels)}
    print("\nDataset counts:", counts)

    min_count = min(counts.values()) if counts else 0
    if min_count < 30:
        print("คำแนะนำ: ควรเก็บอย่างน้อย 80-100 รูปต่อคลาส เพื่อให้ accuracy ดีขึ้น\n")

    stratify = y if min_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify
    )

    model = build_model(num_classes=len(labels))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=0.00005,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    model_path = MODEL_DIR / "thai_digit_model.keras"
    model.save(model_path)
    print(f"Saved model to: {model_path}")

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(
        y_test,
        y_pred,
        target_names=labels,
        digits=4,
        zero_division=0
    )

    report_text = (
        "Thai Handwriting Classification Report\n"
        f"Labels: {labels}\n"
        f"Test Accuracy: {test_acc:.4f}\n\n"
        f"{report}\n"
        f"Dataset counts: {counts}\n"
        "\nNote: This version uses center-crop preprocessing, augmentation, BatchNorm, "
        "Dropout, EarlyStopping, and ReduceLROnPlateau.\n"
    )

    (REPORT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    print("\n" + report)

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    save_training_plot(history)

    print("\nReports saved:")
    print(REPORT_DIR / "classification_report.txt")
    print(REPORT_DIR / "confusion_matrix.png")
    print(REPORT_DIR / "accuracy_plot.png")
    print(REPORT_DIR / "loss_plot.png")


if __name__ == "__main__":
    main()
