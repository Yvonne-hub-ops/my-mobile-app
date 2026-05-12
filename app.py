from flask import Flask, request, jsonify, render_template_string
import cv2
import numpy as np
from PIL import Image
import base64
import io
import json
import math

app = Flask(__name__)

# ─────────────────────────────────────────────
#  MACHINE VISION ANALYSIS ENGINE
# ─────────────────────────────────────────────

def analyze_seed_image(image_bytes):
    """
    Full computer vision pipeline for seed quality assessment.
    Returns structured results with per-metric scores.
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    results = {}

    # ── 1. SEGMENTATION ──────────────────────────────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: treat whole image center region as seed
        cx, cy = w // 2, h // 2
        r = min(w, h) // 3
        cv2.circle(mask, (cx, cy), r, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest = max(contours, key=cv2.contourArea)
    seed_area = cv2.contourArea(largest)
    total_area = h * w

    # ── 2. SIZE & SHAPE ANALYSIS ─────────────────────
    if len(largest) >= 5:
        ellipse = cv2.fitEllipse(largest)
        (ex, ey), (ma, mi), angle = ellipse
        aspect_ratio = min(ma, mi) / max(ma, mi) if max(ma, mi) > 0 else 0
    else:
        x, y, bw, bh = cv2.boundingRect(largest)
        aspect_ratio = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 0
        ma, mi = bw, bh

    # Circularity
    perimeter = cv2.arcLength(largest, True)
    circularity = (4 * math.pi * seed_area / (perimeter ** 2)) if perimeter > 0 else 0
    circularity = min(circularity, 1.0)

    # Convexity (hull solidity)
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = seed_area / hull_area if hull_area > 0 else 0

    size_score = min(100, int((seed_area / total_area) * 500))
    size_score = max(20, size_score)

    shape_score = int((circularity * 0.4 + aspect_ratio * 0.3 + solidity * 0.3) * 100)

    results["size"] = {
        "score": size_score,
        "area_px": int(seed_area),
        "aspect_ratio": round(aspect_ratio, 3),
        "circularity": round(circularity, 3),
        "solidity": round(solidity, 3),
    }
    results["shape"] = {
        "score": shape_score,
        "circularity": round(circularity, 3),
        "solidity": round(solidity, 3),
        "aspect_ratio": round(aspect_ratio, 3),
    }

    # ── 3. COLOR ANALYSIS ────────────────────────────
    seed_mask_bool = mask > 0
    pixels = img_rgb[seed_mask_bool]

    if len(pixels) == 0:
        pixels = img_rgb.reshape(-1, 3)

    mean_color = pixels.mean(axis=0)
    std_color  = pixels.std(axis=0)

    # Hue uniformity via HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hsv_pixels = img_hsv[seed_mask_bool] if seed_mask_bool.sum() > 0 else img_hsv.reshape(-1, 3)
    hue_std = hsv_pixels[:, 0].std()
    sat_mean = hsv_pixels[:, 1].mean()
    val_mean = hsv_pixels[:, 2].mean()

    # Color uniformity score (lower std = more uniform = better)
    color_uniformity = max(0, 100 - int(hue_std * 1.5) - int(std_color.mean() * 0.5))
    color_uniformity = min(100, color_uniformity)

    # Vibrancy (saturation as health proxy)
    vibrancy = int(sat_mean / 255 * 100)

    # Darkness penalty (too dark or too bright)
    brightness = val_mean / 255
    brightness_score = int(100 - abs(brightness - 0.55) * 120)
    brightness_score = max(0, min(100, brightness_score))

    color_score = int(color_uniformity * 0.5 + vibrancy * 0.25 + brightness_score * 0.25)

    results["color"] = {
        "score": color_score,
        "mean_rgb": [int(x) for x in mean_color],
        "hue_std": round(float(hue_std), 2),
        "saturation_mean": round(float(sat_mean), 2),
        "brightness": round(float(brightness), 3),
        "uniformity": color_uniformity,
    }

    # ── 4. TEXTURE & SURFACE ANALYSIS ────────────────
    seed_region = cv2.bitwise_and(gray, gray, mask=mask)
    laplacian_var = cv2.Laplacian(seed_region, cv2.CV_64F).var()

    # GLCM-like roughness via local std
    local_std = cv2.GaussianBlur(
        (gray.astype(np.float32) - cv2.GaussianBlur(gray, (15, 15), 0)) ** 2,
        (5, 5), 0
    )
    roughness = float(np.sqrt(local_std[seed_mask_bool].mean())) if seed_mask_bool.sum() > 0 else 0

    # Ideal: smooth surface → low roughness, moderate sharpness
    texture_smoothness = max(0, 100 - int(roughness * 4))
    texture_score = min(100, max(0, texture_smoothness))

    results["texture"] = {
        "score": texture_score,
        "roughness": round(roughness, 3),
        "laplacian_variance": round(float(laplacian_var), 2),
        "smoothness": texture_smoothness,
    }

    # ── 5. DEFECT DETECTION ──────────────────────────
    # Detect dark spots, discoloration, irregular patches
    seed_only = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    gray_seed = cv2.cvtColor(seed_only, cv2.COLOR_BGR2GRAY)

    # Dark spot detection
    _, dark_spots = cv2.threshold(gray_seed, 40, 255, cv2.THRESH_BINARY_INV)
    dark_spots = cv2.bitwise_and(dark_spots, mask)
    dark_contours, _ = cv2.findContours(dark_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dark_count = len([c for c in dark_contours if cv2.contourArea(c) > 20])
    dark_area_ratio = dark_spots.sum() / 255 / max(seed_area, 1)

    # Edge irregularity
    hull_perimeter = cv2.arcLength(hull, True)
    edge_irregularity = abs(perimeter - hull_perimeter) / hull_perimeter if hull_perimeter > 0 else 0

    defect_penalty = int(dark_count * 8 + dark_area_ratio * 200 + edge_irregularity * 60)
    defect_score = max(0, 100 - defect_penalty)

    results["defects"] = {
        "score": defect_score,
        "dark_spots_count": dark_count,
        "dark_area_ratio": round(float(dark_area_ratio), 4),
        "edge_irregularity": round(float(edge_irregularity), 3),
    }

    # ── 6. OVERALL QUALITY SCORE ─────────────────────
    weights = {"size": 0.15, "shape": 0.25, "color": 0.25, "texture": 0.15, "defects": 0.20}
    overall = sum(results[k]["score"] * weights[k] for k in weights)
    overall = round(overall, 1)

    # Grade
    if overall >= 85:
        grade, label = "A", "Premium"
    elif overall >= 70:
        grade, label = "B", "Good"
    elif overall >= 55:
        grade, label = "C", "Acceptable"
    elif overall >= 40:
        grade, label = "D", "Poor"
    else:
        grade, label = "F", "Reject"

    # ── 7. ANNOTATED IMAGE ───────────────────────────
    annotated = img_rgb.copy()
    cv2.drawContours(annotated, [largest], -1, (0, 220, 100), 2)
    cv2.drawContours(annotated, [hull], -1, (255, 180, 0), 1)

    # Draw bounding ellipse if possible
    if len(largest) >= 5:
        ellipse = cv2.fitEllipse(largest)
        cv2.ellipse(annotated, ellipse, (80, 180, 255), 2)

    # Encode annotated image
    pil_ann = Image.fromarray(annotated)
    buf = io.BytesIO()
    pil_ann.save(buf, format="JPEG", quality=88)
    ann_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "overall_score": overall,
        "grade": grade,
        "grade_label": label,
        "metrics": results,
        "annotated_image": ann_b64,
        "weights": weights,
    }


# ─────────────────────────────────────────────
#  HTML TEMPLATE
# ─────────────────────────────────────────────

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SeedVision AI — Quality Detector</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0b0f0a;
    --bg2: #111710;
    --bg3: #181f16;
    --border: #2a3326;
    --green: #5dde7a;
    --green2: #3cb85c;
    --amber: #e8c547;
    --red: #e85c5c;
    --blue: #5cb8e8;
    --text: #d4e8d0;
    --text2: #7a9470;
    --text3: #4a6444;
    --mono: 'DM Mono', monospace;
    --display: 'Syne', sans-serif;
    --body: 'DM Sans', sans-serif;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--body);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── GRID BACKGROUND ── */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
      linear-gradient(var(--border) 1px, transparent 1px),
      linear-gradient(90deg, var(--border) 1px, transparent 1px);
    background-size: 40px 40px;
    opacity: 0.35;
    pointer-events: none;
    z-index: 0;
  }

  .wrap { position: relative; z-index: 1; max-width: 1100px; margin: 0 auto; padding: 0 24px 80px; }

  /* ── HEADER ── */
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 32px 0 48px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 48px;
  }
  .logo {
    display: flex; align-items: center; gap: 14px;
  }
  .logo-icon {
    width: 44px; height: 44px;
    background: var(--green2);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
  }
  .logo-text h1 {
    font-family: var(--display);
    font-size: 22px; font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--green);
  }
  .logo-text p { font-family: var(--mono); font-size: 11px; color: var(--text3); letter-spacing: 1px; }
  .badge {
    font-family: var(--mono); font-size: 10px;
    background: var(--bg3); border: 1px solid var(--border);
    color: var(--text2); padding: 6px 12px; border-radius: 20px;
    letter-spacing: 0.5px;
  }

  /* ── MAIN GRID ── */
  .main-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }
  @media (max-width: 760px) { .main-grid { grid-template-columns: 1fr; } }

  /* ── PANEL ── */
  .panel {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
  }
  .panel-header {
    display: flex; align-items: center; gap: 10px;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono); font-size: 11px; color: var(--text3);
    letter-spacing: 1px; text-transform: uppercase;
  }
  .panel-header .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); }

  /* ── UPLOAD ZONE ── */
  .upload-zone {
    margin: 24px;
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 48px 24px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
    background: var(--bg3);
    position: relative;
    overflow: hidden;
  }
  .upload-zone:hover, .upload-zone.drag {
    border-color: var(--green2);
    background: rgba(93,222,122,0.04);
  }
  .upload-zone input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .upload-icon { font-size: 42px; margin-bottom: 14px; display: block; }
  .upload-zone h3 { font-family: var(--display); font-size: 16px; font-weight: 700; margin-bottom: 6px; }
  .upload-zone p { font-size: 13px; color: var(--text2); }

  /* ── PREVIEW ── */
  .preview-area { margin: 0 24px 24px; }
  #preview-img, #annotated-img {
    width: 100%; border-radius: 10px;
    border: 1px solid var(--border);
    display: none;
    background: #000;
  }
  .img-label {
    font-family: var(--mono); font-size: 10px; color: var(--text3);
    letter-spacing: 1px; margin-bottom: 8px;
  }

  /* ── ANALYZE BTN ── */
  #analyze-btn {
    display: none;
    margin: 0 24px 24px;
    width: calc(100% - 48px);
    padding: 14px;
    background: var(--green2);
    color: #0b0f0a;
    border: none; border-radius: 10px;
    font-family: var(--display); font-size: 15px; font-weight: 700;
    cursor: pointer; letter-spacing: 0.3px;
    transition: all 0.2s;
  }
  #analyze-btn:hover { background: var(--green); transform: translateY(-1px); }
  #analyze-btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  /* ── SCORE DISPLAY ── */
  .score-big {
    display: flex; align-items: center; justify-content: center;
    flex-direction: column;
    padding: 40px 20px 20px;
  }
  .score-ring {
    position: relative; width: 140px; height: 140px; margin-bottom: 20px;
  }
  .score-ring svg { transform: rotate(-90deg); }
  .score-ring .ring-bg { fill: none; stroke: var(--bg3); stroke-width: 12; }
  .score-ring .ring-val { fill: none; stroke-width: 12; stroke-linecap: round; transition: stroke-dashoffset 1s ease; }
  .score-center {
    position: absolute; inset: 0;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
  }
  .score-num {
    font-family: var(--display); font-size: 38px; font-weight: 800;
    line-height: 1; color: var(--green);
  }
  .score-label { font-family: var(--mono); font-size: 10px; color: var(--text3); letter-spacing: 1px; margin-top: 2px; }
  .grade-badge {
    display: flex; gap: 12px; align-items: center; justify-content: center;
  }
  .grade-letter {
    font-family: var(--display); font-size: 28px; font-weight: 800;
    width: 52px; height: 52px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
  }
  .grade-info h3 { font-family: var(--display); font-size: 18px; font-weight: 700; }
  .grade-info p { font-size: 13px; color: var(--text2); margin-top: 2px; }

  /* ── METRICS ── */
  .metrics-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 12px; padding: 20px 24px;
  }
  .metric-card {
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
  }
  .metric-name {
    font-family: var(--mono); font-size: 10px; color: var(--text3);
    letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px;
    display: flex; align-items: center; justify-content: space-between;
  }
  .metric-score-val {
    font-family: var(--display); font-size: 22px; font-weight: 700;
    line-height: 1;
  }
  .metric-bar-wrap {
    height: 4px; background: var(--border); border-radius: 2px; margin-top: 8px;
  }
  .metric-bar {
    height: 100%; border-radius: 2px; transition: width 0.8s ease;
    width: 0%;
  }
  .metric-detail { font-size: 11px; color: var(--text3); margin-top: 6px; }

  /* ── DETAIL TABLE ── */
  .detail-table { padding: 0 24px 24px; }
  .detail-table table { width: 100%; border-collapse: collapse; }
  .detail-table td {
    padding: 8px 0; font-size: 12px; border-bottom: 1px solid var(--border);
    font-family: var(--mono);
  }
  .detail-table td:first-child { color: var(--text3); }
  .detail-table td:last-child { text-align: right; color: var(--green); }

  /* ── EMPTY STATE ── */
  .empty-state {
    padding: 60px 24px;
    text-align: center;
  }
  .empty-state .icon { font-size: 48px; margin-bottom: 16px; opacity: 0.3; }
  .empty-state p { font-size: 14px; color: var(--text3); font-family: var(--mono); }

  /* ── LOADING ── */
  .spinner {
    display: none; text-align: center; padding: 40px;
  }
  .spinner .dot-row { display: flex; gap: 8px; justify-content: center; margin-bottom: 16px; }
  .spinner .dot-row span {
    width: 8px; height: 8px; border-radius: 50%; background: var(--green);
    animation: bounce 1.2s infinite;
  }
  .spinner .dot-row span:nth-child(2) { animation-delay: 0.2s; }
  .spinner .dot-row span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-12px)} }
  .spinner p { font-family: var(--mono); font-size: 12px; color: var(--text3); letter-spacing: 1px; }

  /* ── BOTTOM ROW ── */
  .bottom-row { margin-top: 24px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  @media (max-width: 760px) { .bottom-row { grid-template-columns: 1fr; } }

  /* ── HISTORY ── */
  .history-list { padding: 16px 24px; }
  .history-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 12px; border-radius: 8px; margin-bottom: 8px;
    background: var(--bg3); border: 1px solid var(--border);
    font-size: 12px; font-family: var(--mono);
    cursor: pointer; transition: border-color 0.15s;
  }
  .history-item:hover { border-color: var(--green2); }
  .hi-score { font-weight: 600; }
  .hi-grade { width: 24px; height: 24px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 11px; }

  /* ── COLORS BY GRADE ── */
  .grade-A { background: rgba(93,222,122,0.15); color: var(--green); border: 1px solid rgba(93,222,122,0.3); }
  .grade-B { background: rgba(93,184,232,0.15); color: var(--blue); border: 1px solid rgba(93,184,232,0.3); }
  .grade-C { background: rgba(232,197,71,0.15); color: var(--amber); border: 1px solid rgba(232,197,71,0.3); }
  .grade-D { background: rgba(232,92,92,0.15); color: var(--red); border: 1px solid rgba(232,92,92,0.3); }
  .grade-F { background: rgba(232,92,92,0.2); color: var(--red); border: 1px solid rgba(232,92,92,0.4); }

  .no-results { display: block; }
</style>
</head>
<body>
<div class="wrap">

  <!-- HEADER -->
  <header>
    <div class="logo">
      <div class="logo-icon">🌱</div>
      <div class="logo-text">
        <h1>SeedVision AI</h1>
        <p>MACHINE VISION QUALITY DETECTOR</p>
      </div>
    </div>
    <div class="badge">CV PIPELINE v2.1</div>
  </header>

  <!-- MAIN GRID -->
  <div class="main-grid">

    <!-- LEFT: Input -->
    <div>
      <div class="panel">
        <div class="panel-header"><div class="dot"></div>INPUT — SEED IMAGE</div>

        <div class="upload-zone" id="drop-zone">
          <input type="file" id="file-input" accept="image/*">
          <span class="upload-icon">📷</span>
          <h3>Drop seed image here</h3>
          <p>or click to browse · JPG, PNG, WEBP</p>
        </div>

        <div class="preview-area">
          <div class="img-label">ORIGINAL IMAGE</div>
          <img id="preview-img" alt="preview">
        </div>

        <button id="analyze-btn">⚡ Analyze Seed Quality</button>
      </div>
    </div>

    <!-- RIGHT: Score -->
    <div>
      <div class="panel" id="results-panel">
        <div class="panel-header"><div class="dot"></div>ANALYSIS RESULT</div>

        <div class="empty-state" id="empty-state">
          <div class="icon">🔬</div>
          <p>Upload a seed image to begin analysis</p>
        </div>

        <div class="spinner" id="spinner">
          <div class="dot-row"><span></span><span></span><span></span></div>
          <p>RUNNING CV PIPELINE...</p>
        </div>

        <div id="result-content" style="display:none">
          <div class="score-big">
            <div class="score-ring">
              <svg width="140" height="140" viewBox="0 0 140 140">
                <circle class="ring-bg" cx="70" cy="70" r="54"/>
                <circle class="ring-val" id="ring-circle" cx="70" cy="70" r="54"
                  stroke-dasharray="339.3" stroke-dashoffset="339.3"/>
              </svg>
              <div class="score-center">
                <div class="score-num" id="score-num">0</div>
                <div class="score-label">SCORE</div>
              </div>
            </div>
            <div class="grade-badge">
              <div class="grade-letter" id="grade-letter">–</div>
              <div class="grade-info">
                <h3 id="grade-label">–</h3>
                <p id="grade-desc">Awaiting analysis</p>
              </div>
            </div>
          </div>

          <div class="metrics-grid" id="metrics-grid"></div>

          <!-- Annotated image -->
          <div class="preview-area" style="margin:0 24px 24px">
            <div class="img-label">ANNOTATED OUTPUT</div>
            <img id="annotated-img" alt="annotated">
          </div>
        </div>
      </div>
    </div>
  </div><!-- /main-grid -->

  <!-- BOTTOM ROW -->
  <div class="bottom-row">

    <!-- Detail table -->
    <div class="panel">
      <div class="panel-header"><div class="dot"></div>DETAILED METRICS</div>
      <div class="detail-table" id="detail-table">
        <div class="empty-state" style="padding:30px 0">
          <p style="font-size:11px">No data yet</p>
        </div>
      </div>
    </div>

    <!-- History -->
    <div class="panel">
      <div class="panel-header"><div class="dot"></div>SCAN HISTORY</div>
      <div class="history-list" id="history-list">
        <div class="empty-state" style="padding:30px 0">
          <p style="font-size:11px">No scans recorded</p>
        </div>
      </div>
    </div>

  </div>

</div><!-- /wrap -->

<script>
  const fileInput  = document.getElementById('file-input');
  const dropZone   = document.getElementById('drop-zone');
  const previewImg = document.getElementById('preview-img');
  const annImg     = document.getElementById('annotated-img');
  const analyzeBtn = document.getElementById('analyze-btn');
  const emptyState = document.getElementById('empty-state');
  const spinner    = document.getElementById('spinner');
  const resultContent = document.getElementById('result-content');

  let currentFile = null;
  let scanHistory = [];

  // ── FILE HANDLING ──────────────────────────────────
  fileInput.addEventListener('change', e => handleFile(e.target.files[0]));
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('drag');
    handleFile(e.dataTransfer.files[0]);
  });

  function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) return;
    currentFile = file;
    const reader = new FileReader();
    reader.onload = ev => {
      previewImg.src = ev.target.result;
      previewImg.style.display = 'block';
    };
    reader.readAsDataURL(file);
    analyzeBtn.style.display = 'block';
    resultContent.style.display = 'none';
    emptyState.style.display = 'block';
    annImg.style.display = 'none';
  }

  // ── ANALYZE ───────────────────────────────────────
  analyzeBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    analyzeBtn.disabled = true;
    emptyState.style.display = 'none';
    resultContent.style.display = 'none';
    spinner.style.display = 'block';

    const fd = new FormData();
    fd.append('image', currentFile);

    try {
      const res  = await fetch('/analyze', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      renderResults(data);
      addHistory(data, currentFile.name);
    } catch (err) {
      alert('Analysis failed: ' + err.message);
    } finally {
      spinner.style.display = 'none';
      analyzeBtn.disabled = false;
    }
  });

  // ── RENDER RESULTS ────────────────────────────────
  function renderResults(data) {
    resultContent.style.display = 'block';

    const score = data.overall_score;
    const grade = data.grade;

    // Ring animation
    const circumference = 339.3;
    const offset = circumference - (score / 100) * circumference;
    const ring = document.getElementById('ring-circle');
    const color = gradeColor(grade);
    ring.style.stroke = color;
    setTimeout(() => { ring.style.strokeDashoffset = offset; }, 50);

    // Score number
    animateNum('score-num', score);

    // Grade badge
    const gl = document.getElementById('grade-letter');
    gl.textContent = grade;
    gl.className = 'grade-letter grade-' + grade;
    document.getElementById('grade-label').textContent = data.grade_label + ' Quality';
    document.getElementById('grade-desc').textContent = gradeDesc(grade);

    // Metrics
    const metricDefs = [
      { key: 'size',    icon: '📐', label: 'Size' },
      { key: 'shape',   icon: '⭕', label: 'Shape' },
      { key: 'color',   icon: '🎨', label: 'Color' },
      { key: 'texture', icon: '🔍', label: 'Texture' },
      { key: 'defects', icon: '🛡', label: 'Defect-Free' },
    ];
    const grid = document.getElementById('metrics-grid');
    grid.innerHTML = '';
    metricDefs.forEach(def => {
      const m = data.metrics[def.key];
      const s = m.score;
      const c = scoreColor(s);
      grid.innerHTML += `
        <div class="metric-card">
          <div class="metric-name">
            <span>${def.icon} ${def.label}</span>
            <span style="color:${c};font-size:9px">${s >= 70 ? '▲' : s >= 40 ? '●' : '▼'}</span>
          </div>
          <div class="metric-score-val" style="color:${c}">${s}</div>
          <div class="metric-bar-wrap">
            <div class="metric-bar" id="bar-${def.key}" style="background:${c}"></div>
          </div>
          <div class="metric-detail">${metricDetail(def.key, m)}</div>
        </div>`;
    });
    setTimeout(() => {
      metricDefs.forEach(def => {
        const bar = document.getElementById('bar-' + def.key);
        if (bar) bar.style.width = data.metrics[def.key].score + '%';
      });
    }, 100);

    // Annotated image
    annImg.src = 'data:image/jpeg;base64,' + data.annotated_image;
    annImg.style.display = 'block';

    // Detail table
    renderDetailTable(data);
  }

  function metricDetail(key, m) {
    if (key === 'size')    return `Area: ${m.area_px.toLocaleString()} px²`;
    if (key === 'shape')   return `Circularity: ${(m.circularity*100).toFixed(1)}% · Solidity: ${(m.solidity*100).toFixed(1)}%`;
    if (key === 'color')   return `RGB(${m.mean_rgb.join(',')}) · Brightness: ${(m.brightness*100).toFixed(0)}%`;
    if (key === 'texture') return `Roughness: ${m.roughness.toFixed(2)} · Smoothness: ${m.smoothness}%`;
    if (key === 'defects') return `Spots: ${m.dark_spots_count} · Edge dev: ${(m.edge_irregularity*100).toFixed(1)}%`;
    return '';
  }

  function renderDetailTable(data) {
    const m = data.metrics;
    const rows = [
      ['Overall Score', data.overall_score + ' / 100'],
      ['Grade', data.grade + ' — ' + data.grade_label],
      ['Seed Area (px²)', m.size.area_px.toLocaleString()],
      ['Aspect Ratio', m.size.aspect_ratio],
      ['Circularity', (m.shape.circularity * 100).toFixed(1) + '%'],
      ['Solidity', (m.shape.solidity * 100).toFixed(1) + '%'],
      ['Color Uniformity', m.color.uniformity + '%'],
      ['Mean Brightness', (m.color.brightness * 100).toFixed(1) + '%'],
      ['Surface Roughness', m.texture.roughness.toFixed(3)],
      ['Dark Spots', m.defects.dark_spots_count],
      ['Dark Area Ratio', (m.defects.dark_area_ratio * 100).toFixed(2) + '%'],
      ['Edge Irregularity', (m.defects.edge_irregularity * 100).toFixed(1) + '%'],
    ];
    let html = '<table>';
    rows.forEach(([k, v]) => { html += `<tr><td>${k}</td><td>${v}</td></tr>`; });
    html += '</table>';
    document.getElementById('detail-table').innerHTML = html;
  }

  function addHistory(data, filename) {
    const ts = new Date().toLocaleTimeString();
    scanHistory.unshift({ score: data.overall_score, grade: data.grade, label: data.grade_label, file: filename, ts });
    if (scanHistory.length > 8) scanHistory.pop();
    const list = document.getElementById('history-list');
    list.innerHTML = scanHistory.map(s => `
      <div class="history-item">
        <div>
          <div style="color:var(--text);margin-bottom:2px">${s.file.substring(0,22)}</div>
          <div style="color:var(--text3);font-size:10px">${s.ts}</div>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <span class="hi-score" style="color:${gradeColor(s.grade)}">${s.score}</span>
          <div class="hi-grade grade-${s.grade}">${s.grade}</div>
        </div>
      </div>`).join('');
  }

  // ── HELPERS ───────────────────────────────────────
  function gradeColor(g) {
    return {A:'#5dde7a',B:'#5cb8e8',C:'#e8c547',D:'#e88c5c',F:'#e85c5c'}[g] || '#5dde7a';
  }
  function scoreColor(s) {
    if (s >= 80) return '#5dde7a';
    if (s >= 60) return '#a8d85d';
    if (s >= 40) return '#e8c547';
    return '#e85c5c';
  }
  function gradeDesc(g) {
    return {
      A: 'Excellent germination potential',
      B: 'Good viability, minor imperfections',
      C: 'Acceptable, moderate quality',
      D: 'Low quality, not recommended',
      F: 'Reject — significant defects'
    }[g] || '';
  }
  function animateNum(id, target) {
    const el = document.getElementById(id);
    let cur = 0;
    const step = target / 40;
    const t = setInterval(() => {
      cur = Math.min(cur + step, target);
      el.textContent = Math.round(cur);
      if (cur >= target) clearInterval(t);
    }, 20);
  }
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    try:
        image_bytes = file.read()
        result = analyze_seed_image(image_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)
