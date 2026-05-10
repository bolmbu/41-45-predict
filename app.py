from pathlib import Path
import base64
import io
import json
import time
import shutil
from functools import wraps

import numpy as np
from PIL import Image, ImageOps
from flask import Flask, jsonify, render_template, request, session, redirect, url_for, flash
import tensorflow as tf


# =========================================================
# Path หลักของโปรเจกต์
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "thai_digit_model.keras"
LABELS_PATH = BASE_DIR / "labels.json"


# =========================================================
# Flask App
# =========================================================
app = Flask(__name__)
app.secret_key = "thai-handwriting-demo-secret-key"


# =========================================================
# Admin Login
# =========================================================
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "1234"


# =========================================================
# Load labels
# =========================================================
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = json.load(f)


# =========================================================
# Global model cache
# =========================================================
model = None
model_mtime = None
model_path_loaded = None


# =========================================================
# Folder setup
# =========================================================
def ensure_folders():
    DATASET_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    (BASE_DIR / "reports").mkdir(exist_ok=True)

    for label in LABELS:
        (DATASET_DIR / label).mkdir(parents=True, exist_ok=True)


# =========================================================
# Admin helpers
# =========================================================
def is_admin_logged_in():
    return session.get("admin_logged_in", False) is True


def admin_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if not is_admin_logged_in():
            flash("กรุณาเข้าสู่ระบบแอดมินก่อน", "warning")
            return redirect(url_for("admin_login"))
        return view_func(*args, **kwargs)

    return wrapper


# =========================================================
# Model loading
# =========================================================
def get_latest_model_path():
    """
    เลือกโมเดลล่าสุดจากโฟลเดอร์ models/ อัตโนมัติ
    รองรับ .keras และ .h5

    ตัวอย่าง:
    models/thai_digit_model.keras
    models/thai_digit_model_acc_0.8367.keras
    models/model_v2.h5

    ระบบจะเลือกไฟล์ที่แก้ไขล่าสุด
    """
    MODEL_DIR.mkdir(exist_ok=True)

    model_files = []
    for ext in ("*.keras", "*.h5"):
        model_files.extend(MODEL_DIR.glob(ext))

    if not model_files:
        return None

    return max(model_files, key=lambda p: p.stat().st_mtime)


def load_model_if_needed():
    """
    โหลดโมเดลเฉพาะตอนจำเป็น:
    - ยังไม่เคยโหลด
    - มีไฟล์โมเดลใหม่กว่าเดิม
    - เปลี่ยนไปใช้ไฟล์โมเดลคนละชื่อ
    """
    global model, model_mtime, model_path_loaded

    latest_model_path = get_latest_model_path()

    if latest_model_path is None:
        model = None
        model_mtime = None
        model_path_loaded = None
        return None

    current_mtime = latest_model_path.stat().st_mtime

    if (
        model is None
        or model_mtime != current_mtime
        or model_path_loaded != latest_model_path
    ):
        print(f"Loading model: {latest_model_path.name}")
        model = tf.keras.models.load_model(latest_model_path)
        model_mtime = current_mtime
        model_path_loaded = latest_model_path

    return model


def warmup_model():
    """
    โหลดโมเดลล่าสุด และ predict รูปว่าง 1 ครั้งตอนเปิด server
    ช่วยให้ Predict ครั้งแรกไม่ช้ามาก
    """
    current_model = load_model_if_needed()

    if current_model is None:
        print("No model found yet. Train or upload model first.")
        return

    try:
        dummy = np.zeros((1, 32, 32, 1), dtype="float32")
        current_model.predict(dummy, verbose=0)
        print("Model loaded and warmed up.")
    except Exception as e:
        print("Warmup failed:", e)
        print("Server ยังรันได้ แต่ให้เช็กว่า app.py กับ train_model.py ใช้ขนาด input ตรงกันหรือไม่")


# =========================================================
# Image preprocessing
# =========================================================
def decode_base64_image(data_url: str) -> Image.Image:
    """
    รับภาพจาก canvas ที่เป็น base64 แล้วแปลงเป็น PIL grayscale image
    """
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # รวมภาพกับพื้นหลังสีขาว กันปัญหา transparency
    bg = Image.new("RGBA", img.size, "WHITE")
    bg.alpha_composite(img)

    return bg.convert("L")


def center_crop_handwriting(img: Image.Image) -> Image.Image:
    """
    preprocessing ต้องตรงกับ train_model.py

    ขั้นตอน:
    1. grayscale
    2. invert ให้ลายมือเป็นสีขาว พื้นหลังดำ
    3. หา bounding box ของเส้นลายมือ
    4. crop เฉพาะตัวเลข
    5. pad ให้เป็น square
    6. resize เป็น 32x32
    """
    img = img.convert("L")
    img = ImageOps.invert(img)

    arr = np.array(img)

    # เลือก pixel ที่น่าจะเป็นเส้นลายมือ
    mask = arr > 30

    # ถ้าไม่มีเส้นเลย ให้ resize ปกติ
    if not mask.any():
        return img.resize((32, 32))

    ys, xs = np.where(mask)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # เพิ่ม margin กัน crop ชิดเกินไป
    margin = 12
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(arr.shape[1] - 1, x2 + margin)
    y2 = min(arr.shape[0] - 1, y2 + margin)

    crop = img.crop((x1, y1, x2 + 1, y2 + 1))

    # pad เป็นสี่เหลี่ยมจัตุรัส
    w, h = crop.size
    side = max(w, h)

    square = Image.new("L", (side, side), 0)
    square.paste(crop, ((side - w) // 2, (side - h) // 2))

    # เพิ่มขอบเล็กน้อยก่อน resize
    pad = max(4, side // 12)
    padded = Image.new("L", (side + pad * 2, side + pad * 2), 0)
    padded.paste(square, (pad, pad))

    return padded.resize((32, 32))


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    แปลงภาพให้เป็น input สำหรับโมเดล
    output shape = (1, 32, 32, 1)
    """
    img = center_crop_handwriting(img)
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 32, 32, 1)
    return arr


# =========================================================
# Dataset helpers
# =========================================================
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def image_files_in(label_dir: Path):
    """
    นับเฉพาะไฟล์รูปที่ใช้ train ได้จริง
    ไม่เอา .DS_Store, .gitkeep หรือไฟล์อื่น
    """
    files = []

    for ext in IMAGE_EXTENSIONS:
        files.extend(label_dir.glob(f"*{ext}"))
        files.extend(label_dir.glob(f"*{ext.upper()}"))

    return sorted(set(files), key=lambda p: p.stat().st_mtime)


def dataset_counts():
    """
    นับจำนวนรูปในแต่ละ class
    """
    ensure_folders()

    counts = {}

    for label in LABELS:
        label_dir = DATASET_DIR / label
        counts[label] = len(image_files_in(label_dir))

    return counts


# =========================================================
# Global template variables
# =========================================================
@app.context_processor
def inject_globals():
    return {
        "admin_logged_in": is_admin_logged_in(),
        "admin_username": ADMIN_USERNAME,
    }


# =========================================================
# Pages
# =========================================================
@app.route("/")
def index():
    latest_model_path = get_latest_model_path()

    return render_template(
        "index.html",
        labels=LABELS,
        has_model=latest_model_path is not None
    )


@app.route("/collect")
def collect():
    """
    ย้าย collect dataset ไปอยู่ใน Admin Dashboard
    """
    if is_admin_logged_in():
        return redirect(url_for("admin_dashboard"))

    flash("ส่วนเก็บ Dataset อยู่ในหน้า Admin", "warning")
    return redirect(url_for("admin_login"))


@app.route("/admin")
def admin_root():
    if is_admin_logged_in():
        return redirect(url_for("admin_dashboard"))

    return redirect(url_for("admin_login"))


@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            session["admin_user"] = username
            flash("เข้าสู่ระบบสำเร็จ", "success")
            return redirect(url_for("admin_dashboard"))

        flash("ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง", "danger")

    return render_template("admin_login.html")


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    latest_model_path = get_latest_model_path()

    return render_template(
        "admin_dashboard.html",
        labels=LABELS,
        has_model=latest_model_path is not None,
        latest_model_name=(latest_model_path.name if latest_model_path else None),
        counts=dataset_counts(),
    )


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    flash("ออกจากระบบแล้ว", "success")
    return redirect(url_for("admin_login"))


# =========================================================
# APIs
# =========================================================
@app.route("/api/counts")
def api_counts():
    return jsonify({
        "ok": True,
        "counts": dataset_counts()
    })


@app.route("/api/save-sample", methods=["POST"])
def save_sample():
    ensure_folders()

    data = request.get_json(force=True)

    label = data.get("label")
    image_data = data.get("image")

    if label not in LABELS:
        return jsonify({
            "ok": False,
            "error": "Invalid label"
        }), 400

    if not image_data:
        return jsonify({
            "ok": False,
            "error": "No image data"
        }), 400

    img = decode_base64_image(image_data)

    label_dir = DATASET_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)

    # นับจากไฟล์รูปจริงทั้งหมด
    current_count = len(image_files_in(label_dir)) + 1

    filename = f"{label}_{current_count:04d}_{timestamp}.png"
    img.save(label_dir / filename)

    return jsonify({
        "ok": True,
        "filename": filename,
        "counts": dataset_counts()
    })


@app.route("/api/delete-last", methods=["POST"])
def delete_last():
    data = request.get_json(force=True)

    label = data.get("label")

    if label not in LABELS:
        return jsonify({
            "ok": False,
            "error": "Invalid label"
        }), 400

    label_dir = DATASET_DIR / label
    files = image_files_in(label_dir)

    if not files:
        return jsonify({
            "ok": False,
            "error": "No file to delete"
        }), 404

    last_file = files[-1]
    last_file.unlink()

    return jsonify({
        "ok": True,
        "deleted": last_file.name,
        "counts": dataset_counts()
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    current_model = load_model_if_needed()

    if current_model is None:
        return jsonify({
            "ok": False,
            "error": "ยังไม่มีโมเดล กรุณาเก็บ dataset แล้วรัน python train_model.py ก่อน"
        }), 400

    data = request.get_json(force=True)
    image_data = data.get("image")

    if not image_data:
        return jsonify({
            "ok": False,
            "error": "No image data"
        }), 400

    img = decode_base64_image(image_data)
    x = preprocess_image(img)

    probs = current_model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))

    top = []

    for i, p in enumerate(probs):
        top.append({
            "label": LABELS[i],
            "confidence": float(p)
        })

    top.sort(key=lambda item: item["confidence"], reverse=True)

    return jsonify({
        "ok": True,
        "prediction": LABELS[idx],
        "confidence": float(probs[idx]),
        "top": top
    })


@app.route("/api/upload-model", methods=["POST"])
@admin_required
def upload_model():
    """
    Upload model ใหม่ผ่านหน้า Admin

    หมายเหตุ:
    - ระบบยัง save ไฟล์ upload เป็นชื่อหลัก thai_digit_model.keras
    - แต่ตอนรัน server ระบบจะเลือกไฟล์ล่าสุดใน models/ อัตโนมัติ
    """
    global model, model_mtime, model_path_loaded

    ensure_folders()

    if "model" not in request.files:
        return jsonify({
            "ok": False,
            "error": "No model file"
        }), 400

    file = request.files["model"]

    if file.filename == "":
        return jsonify({
            "ok": False,
            "error": "No selected file"
        }), 400

    allowed = [".keras", ".h5"]
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed:
        return jsonify({
            "ok": False,
            "error": "รองรับเฉพาะ .keras หรือ .h5"
        }), 400

    backup_dir = MODEL_DIR / "backup"
    backup_dir.mkdir(exist_ok=True)

    if DEFAULT_MODEL_PATH.exists():
        backup_name = f"thai_digit_model_backup_{int(time.time())}.keras"
        shutil.copy2(DEFAULT_MODEL_PATH, backup_dir / backup_name)

    # save เป็นชื่อหลัก เพื่อให้ใช้ง่าย
    file.save(DEFAULT_MODEL_PATH)

    # บังคับให้โหลดใหม่ตอน Predict ครั้งถัดไป
    model = None
    model_mtime = None
    model_path_loaded = None

    return jsonify({
        "ok": True,
        "message": "อัปโหลดโมเดลสำเร็จ"
    })


# =========================================================
# Run server
# =========================================================
if __name__ == "__main__":
    ensure_folders()
    warmup_model()

    # ใช้ 127.0.0.1 เพื่อให้เข้าได้เฉพาะเครื่องตัวเอง ปลอดภัยกว่า
    # debug=False ทำให้รันนิ่งและ Predict เร็วกว่า debug=True
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )