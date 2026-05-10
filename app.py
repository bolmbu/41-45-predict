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

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "thai_digit_model.keras"
LABELS_PATH = BASE_DIR / "labels.json"

app = Flask(__name__)
app.secret_key = "thai-handwriting-demo-secret-key"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "1234"

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    LABELS = json.load(f)

model = None
model_mtime = None


def ensure_folders():
    DATASET_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    (BASE_DIR / "reports").mkdir(exist_ok=True)
    for label in LABELS:
        (DATASET_DIR / label).mkdir(parents=True, exist_ok=True)


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


def load_model_if_needed():
    global model, model_mtime

    if not MODEL_PATH.exists():
        model = None
        model_mtime = None
        return None

    current_mtime = MODEL_PATH.stat().st_mtime
    if model is None or model_mtime != current_mtime:
        model = tf.keras.models.load_model(MODEL_PATH)
        model_mtime = current_mtime

    return model


def decode_base64_image(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    img_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    bg = Image.new("RGBA", img.size, "WHITE")
    bg.alpha_composite(img)
    return bg.convert("L")


def center_crop_handwriting(img: Image.Image) -> Image.Image:
    """
    ต้องเหมือนกับ preprocessing ใน train_model.py
    เพื่อให้ตอน Predict ใช้รูปแบบข้อมูลเดียวกับตอน Train
    """
    img = img.convert("L")
    img = ImageOps.invert(img)

    arr = np.array(img)
    mask = arr > 30

    if not mask.any():
        return img.resize((32, 32))

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

    return padded.resize((32, 32))


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = center_crop_handwriting(img)
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 32, 32, 1)
    return arr



IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def image_files_in(label_dir: Path):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(label_dir.glob(f"*{ext}"))
        files.extend(label_dir.glob(f"*{ext.upper()}"))
    return sorted(set(files), key=lambda p: p.stat().st_mtime)


def dataset_counts():
    """
    นับรูปทั้งหมดที่ train_model.py ใช้ได้จริง
    รองรับ .png, .jpg, .jpeg, .webp, .bmp ทั้งตัวพิมพ์เล็ก/ใหญ่
    ไม่ได้นับ .DS_Store, .gitkeep หรือไฟล์อื่นที่ไม่ใช่รูป
    """
    ensure_folders()
    counts = {}
    for label in LABELS:
        label_dir = DATASET_DIR / label
        counts[label] = len(image_files_in(label_dir))
    return counts


@app.context_processor
def inject_globals():
    return {
        "admin_logged_in": is_admin_logged_in(),
        "admin_username": ADMIN_USERNAME,
    }


@app.route("/")
def index():
    return render_template("index.html", labels=LABELS, has_model=MODEL_PATH.exists())


@app.route("/collect")
def collect():
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
    return render_template(
        "admin_dashboard.html",
        labels=LABELS,
        has_model=MODEL_PATH.exists(),
        counts=dataset_counts(),
    )


@app.route("/admin/logout")
def admin_logout():
    session.clear()
    flash("ออกจากระบบแล้ว", "success")
    return redirect(url_for("admin_login"))


@app.route("/api/counts")
def api_counts():
    return jsonify({"ok": True, "counts": dataset_counts()})


@app.route("/api/save-sample", methods=["POST"])
def save_sample():
    ensure_folders()
    data = request.get_json(force=True)
    label = data.get("label")
    image_data = data.get("image")

    if label not in LABELS:
        return jsonify({"ok": False, "error": "Invalid label"}), 400
    if not image_data:
        return jsonify({"ok": False, "error": "No image data"}), 400

    img = decode_base64_image(image_data)
    label_dir = DATASET_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time() * 1000)
    current_count = len(list(label_dir.glob("*.png"))) + 1
    filename = f"{label}_{current_count:04d}_{timestamp}.png"
    img.save(label_dir / filename)

    return jsonify({"ok": True, "filename": filename, "counts": dataset_counts()})


@app.route("/api/delete-last", methods=["POST"])
def delete_last():
    data = request.get_json(force=True)
    label = data.get("label")
    if label not in LABELS:
        return jsonify({"ok": False, "error": "Invalid label"}), 400

    label_dir = DATASET_DIR / label
    files = image_files_in(label_dir)
    if not files:
        return jsonify({"ok": False, "error": "No file to delete"}), 404

    last_file = files[-1]
    last_file.unlink()
    return jsonify({"ok": True, "deleted": last_file.name, "counts": dataset_counts()})


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
        return jsonify({"ok": False, "error": "No image data"}), 400

    img = decode_base64_image(image_data)
    x = preprocess_image(img)
    probs = current_model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))

    top = []
    for i, p in enumerate(probs):
        top.append({"label": LABELS[i], "confidence": float(p)})
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
    global model, model_mtime
    ensure_folders()

    if "model" not in request.files:
        return jsonify({"ok": False, "error": "No model file"}), 400

    file = request.files["model"]
    if file.filename == "":
        return jsonify({"ok": False, "error": "No selected file"}), 400

    allowed = [".keras", ".h5"]
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        return jsonify({"ok": False, "error": "รองรับเฉพาะ .keras หรือ .h5"}), 400

    backup_dir = MODEL_DIR / "backup"
    backup_dir.mkdir(exist_ok=True)
    if MODEL_PATH.exists():
        backup_name = f"thai_digit_model_backup_{int(time.time())}.keras"
        shutil.copy2(MODEL_PATH, backup_dir / backup_name)

    file.save(MODEL_PATH)
    model = None
    model_mtime = None
    return jsonify({"ok": True, "message": "อัปโหลดโมเดลสำเร็จ"})


if __name__ == "__main__":
    ensure_folders()
    app.run(host="0.0.0.0", port=5000, debug=True)
