const canvas = document.getElementById("drawCanvas");
let ctx = null;
let drawing = false;

if (canvas) {
  ctx = canvas.getContext("2d");
  ctx.lineWidth = 18;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#0f172a";

  canvas.addEventListener("mousedown", startDraw);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("mouseup", stopDraw);
  canvas.addEventListener("mouseleave", stopDraw);
  canvas.addEventListener("touchstart", startDraw, { passive: false });
  canvas.addEventListener("touchmove", draw, { passive: false });
  canvas.addEventListener("touchend", stopDraw);
}

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const point = e.touches ? e.touches[0] : e;
  return {
    x: (point.clientX - rect.left) * (canvas.width / rect.width),
    y: (point.clientY - rect.top) * (canvas.height / rect.height),
  };
}

function startDraw(e) {
  e.preventDefault();
  drawing = true;
  const p = getPos(e);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
}

function draw(e) {
  if (!drawing) return;
  e.preventDefault();
  const p = getPos(e);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
}

function stopDraw() { drawing = false; }

function clearCanvas() {
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const pred = document.getElementById("prediction");
  const conf = document.getElementById("confidence");
  const topList = document.getElementById("topList");
  if (pred) pred.textContent = "-";
  if (conf) conf.textContent = "Confidence: -";
  if (topList) topList.innerHTML = "";
}

function canvasImage() { return canvas.toDataURL("image/png"); }

async function predictDigit() {
  const status = document.getElementById("status");
  const pred = document.getElementById("prediction");
  const conf = document.getElementById("confidence");
  const topList = document.getElementById("topList");

  status.textContent = "กำลังทำนาย...";
  const res = await fetch("/api/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ image: canvasImage() }),
  });

  const data = await res.json();
  if (!data.ok) {
    status.textContent = data.error || "Predict failed";
    return;
  }

  pred.textContent = data.prediction;
  conf.textContent = "Confidence: " + (data.confidence * 100).toFixed(2) + "%";
  status.textContent = "ทำนายสำเร็จ";

  if (topList && data.top) {
    topList.innerHTML = data.top.map(item =>
      `<div class="top-item"><strong>${item.label}</strong><span>${(item.confidence * 100).toFixed(2)}%</span></div>`
    ).join("");
  }
}

async function saveSample() {
  const label = document.getElementById("labelSelect").value;
  const status = document.getElementById("status");
  status.textContent = "กำลังบันทึก...";

  const res = await fetch("/api/save-sample", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ label: label, image: canvasImage() }),
  });
  const data = await res.json();
  if (!data.ok) {
    status.textContent = data.error || "Save failed";
    return;
  }

  status.textContent = "บันทึกแล้ว: " + data.filename;
  updateCounts(data.counts);
  clearCanvas();
}

async function deleteLast() {
  const label = document.getElementById("labelSelect").value;
  const status = document.getElementById("status");

  const res = await fetch("/api/delete-last", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ label: label }),
  });
  const data = await res.json();
  if (!data.ok) {
    status.textContent = data.error || "Delete failed";
    return;
  }

  status.textContent = "ลบรูปล่าสุดแล้ว: " + data.deleted;
  updateCounts(data.counts);
}

function updateCounts(counts) {
  for (const [label, count] of Object.entries(counts)) {
    const el = document.getElementById("count-" + label);
    if (el) el.textContent = count;
  }
}

async function uploadModel() {
  const input = document.getElementById("modelFile");
  const status = document.getElementById("uploadStatus");
  if (!input.files.length) {
    status.textContent = "กรุณาเลือกไฟล์โมเดลก่อน";
    return;
  }

  const formData = new FormData();
  formData.append("model", input.files[0]);
  status.textContent = "กำลังอัปโหลด...";

  const res = await fetch("/api/upload-model", { method: "POST", body: formData });
  const data = await res.json();
  if (!data.ok) {
    status.textContent = data.error || "Upload failed";
    return;
  }
  status.textContent = data.message;
}
