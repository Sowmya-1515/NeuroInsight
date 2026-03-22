"""
NeuroInsight — Flask API Backend
=================================
Connects the trained Phase II ResNet model + FAISS index to the web frontend.

Endpoints:
  POST /api/predict          → upload MRI image, get classification + retrieval results
  POST /api/report           → generate a polished PDF report from a result JSON
  GET  /api/health           → health check
  GET  /api/stats            → index statistics
"""

import os
import json
import time
import base64
import numpy as np
import pandas as pd
import torch
import faiss

from generate import generate_pdf_report
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from torchvision import transforms, models
from PIL import Image
from io import BytesIO
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# CONFIG
# FIX: Use dynamic BASE_DIR instead of hardcoded absolute path
# ============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "training_output_enhanced", "best_model.pth")

# Download model from Hugging Face if not present locally
HF_MODEL_URL = "https://huggingface.co/Sowmyas15/NeuroInsight-model/resolve/main/best_model.pth"
if not os.path.exists(MODEL_PATH):
    import urllib.request
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("[NeuroInsight] Downloading model from Hugging Face...")
    urllib.request.urlretrieve(HF_MODEL_URL, MODEL_PATH)
    print("[NeuroInsight] Model downloaded successfully!")

FAISS_DIR      = os.path.join(BASE_DIR, "faiss_index")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings_output")

EMBED_DIM   = 128
IMG_SIZE    = 160
TOP_K       = 50
FINAL_K     = 10
MAX_FILE_MB = 20

WEIGHTS = {
    "grade":    0.35,
    "severity": 0.30,
    "size":     0.20,
    "location": 0.15,
}

GRADE_MAP    = {0: "LGG",      1: "HGG"}
SEVERITY_MAP = {0: "low",      1: "medium",  2: "high"}
SIZE_MAP     = {0: "small",    1: "medium",  2: "large"}
LOCATION_MAP = {0: "left",     1: "right",   2: "bilateral"}


# ============================================================
# MODEL DEFINITION
# ============================================================
class Phase2Model(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=False, dropout_rate=0.3):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.encoder      = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim  = 2048
        self.dropout_rate = dropout_rate

        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, embedding_dim)
        )

        def _head(out):
            return nn.Sequential(
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, out)
            )

        self.grade_head            = _head(2)
        self.severity_head         = _head(3)
        self.size_head             = _head(3)
        self.location_head         = _head(3)
        self.tumor_head            = _head(2)

        self.tumor_confidence_head = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features  = self.encoder(x).view(x.size(0), -1)
        embedding = F.normalize(self.projection_head(features), dim=1)
        return {
            "embedding":        embedding,
            "grade_logits":     self.grade_head(features),
            "severity_logits":  self.severity_head(features),
            "size_logits":      self.size_head(features),
            "location_logits":  self.location_head(features),
            "tumor_logits":     self.tumor_head(features),
            "tumor_confidence": self.tumor_confidence_head(features).squeeze(),
            "features":         features,
        }


# ============================================================
# GRAD-CAM
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, input, output):
            self.activations = output.detach()
        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        outputs      = self.model(input_tensor)
        logits       = outputs["grade_logits"]
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                                mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam


def apply_heatmap(original_img: Image.Image, cam: np.ndarray) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    orig    = np.array(original_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"))
    heat    = cm.jet(cam)[:, :, :3]
    heat    = (heat * 255).astype(np.uint8)
    overlay = (0.55 * orig + 0.45 * heat).astype(np.uint8)
    buf     = BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def img_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# RETRIEVAL ENGINE
# ============================================================
class RetrievalEngine:
    def __init__(self):
        self.device    = self._get_device()
        self.model     = None
        self.grad_cam  = None
        self.index     = None
        self.metadata  = None
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.ready = False
        self._load()

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _load(self):
        print(f"[NeuroScan] Device: {self.device}")

        print("[NeuroInsight] Loading model...")
        self.model = Phase2Model(embedding_dim=EMBED_DIM, pretrained=False).to(self.device)
        missing, unexpected = self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.device), strict=False
        )
        if missing:
            print(f"[NeuroInsight] Missing keys: {missing}")
        if unexpected:
            print(f"[NeuroInsight] Unexpected keys: {unexpected}")
        self.model.eval()

        target_layer  = list(self.model.encoder.children())[-3][-1].conv3
        self.grad_cam = GradCAM(self.model, target_layer)
        print("[NeuroScan] Model loaded ✓")

        faiss_path = os.path.join(FAISS_DIR, "faiss_ivfpq.index")
        print(f"[NeuroScan] Loading FAISS index from {faiss_path}...")
        self.index = faiss.read_index(os.path.join(FAISS_DIR, "faiss_flat.index"))
        print(f"[NeuroScan] FAISS index loaded ✓  ({self.index.ntotal} vectors)")

        meta_path     = os.path.join(EMBEDDINGS_DIR, "metadata.csv")
        self.metadata = pd.read_csv(meta_path)
        print(f"[NeuroScan] Metadata loaded ✓  ({len(self.metadata)} rows)")

        self.ready = True
        print("[NeuroScan] RetrievalEngine ready ✓\n")

    def _softmax_conf(self, logits):
        return float(torch.softmax(logits, dim=1).max().item())

    def _bin_distance(self, a, b, order):
        if a not in order or b not in order:
            return 0.0
        d = abs(order.index(a) - order.index(b))
        return 1.0 if d == 0 else 0.5 if d == 1 else 0.0

    def _attr_similarity(self, query_attrs, row):
        scores = {
            "grade":    1.0 if query_attrs["grade"] == row["true_grade"] else 0.0,
            "severity": self._bin_distance(query_attrs["severity"],
                                           row["true_severity"], ["low","medium","high"]),
            "size":     self._bin_distance(query_attrs["size"],
                                           row["true_size"], ["small","medium","large"]),
            "location": 1.0 if query_attrs["location"] == row["true_location"] else 0.0,
        }
        scores["weighted"] = sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
        return scores

    def predict(self, image_bytes: bytes) -> dict:
        t0 = time.time()

        pil_img    = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)

        grade_idx    = outputs["grade_logits"].argmax(1).item()
        severity_idx = outputs["severity_logits"].argmax(1).item()
        size_idx     = outputs["size_logits"].argmax(1).item()
        location_idx = outputs["location_logits"].argmax(1).item()
        tumor_idx    = outputs["tumor_logits"].argmax(1).item()
        tumor_conf   = (float(outputs["tumor_confidence"].item())
                        if outputs["tumor_confidence"].dim() == 0
                        else float(outputs["tumor_confidence"][0].item()))

        query_attrs = {
            "grade":    GRADE_MAP[grade_idx],
            "severity": SEVERITY_MAP[severity_idx],
            "size":     SIZE_MAP[size_idx],
            "location": LOCATION_MAP[location_idx],
        }
        confidence = {
            "grade":    round(self._softmax_conf(outputs["grade_logits"]) * 100, 1),
            "severity": round(self._softmax_conf(outputs["severity_logits"]) * 100, 1),
            "size":     round(self._softmax_conf(outputs["size_logits"]) * 100, 1),
            "location": round(self._softmax_conf(outputs["location_logits"]) * 100, 1),
        }

        embedding = outputs["embedding"].cpu().numpy().astype("float32")
        norm      = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        D, I = self.index.search(embedding, TOP_K + 1)
        D, I = D[0], I[0]

        candidates = []
        for dist, idx in zip(D, I):
            if idx < 0 or idx >= len(self.metadata):
                continue
            row         = self.metadata.iloc[idx]
            cosine_sim  = float(max(0.0, 1.0 - dist / 2.0))
            attr_scores = self._attr_similarity(query_attrs, row)
            final_score = 0.60 * cosine_sim + 0.40 * attr_scores["weighted"]
            candidates.append({
                "idx":        int(idx),
                "cosine_sim": cosine_sim,
                "attr":       attr_scores,
                "final":      final_score,
                "meta":       row,
            })

        candidates.sort(key=lambda x: x["final"], reverse=True)

        results = []
        for rank, c in enumerate(candidates[:FINAL_K], 1):
            row = c["meta"]
            results.append({
                "rank":          rank,
                "similarity":    round(c["final"] * 100, 1),
                "embedding_sim": round(c["cosine_sim"] * 100, 1),
                "attr_sim":      round(c["attr"]["weighted"] * 100, 1),
                "breakdown": {
                    "grade":    round(c["attr"]["grade"] * 100, 1),
                    "severity": round(c["attr"]["severity"] * 100, 1),
                    "size":     round(c["attr"]["size"] * 100, 1),
                    "location": round(c["attr"]["location"] * 100, 1),
                },
                "diagnosis": {
                    "grade":    row["true_grade"],
                    "severity": row["true_severity"],
                    "size":     row["true_size"],
                    "location": row["true_location"],
                },
                "image_path": row["image_path"],
            })

        cam          = self.grad_cam.generate(img_tensor.clone())
        heatmap_b64  = apply_heatmap(pil_img, cam)
        original_b64 = img_to_base64(pil_img.resize((IMG_SIZE, IMG_SIZE)))
        inference_ms = round((time.time() - t0) * 1000)
        severity_score = {"low": 0.25, "medium": 0.60, "high": 0.92}[query_attrs["severity"]]

        return {
            "status":       "success",
            "inference_ms": inference_ms,
            "diagnosis": {
                "grade":            query_attrs["grade"],
                "severity":         query_attrs["severity"],
                "size":             query_attrs["size"],
                "location":         query_attrs["location"],
                "severity_score":   severity_score,
                "tumor_present":    bool(tumor_idx == 1),
                "tumor_confidence": round(tumor_conf * 100, 1),
            },
            "confidence": confidence,
            "retrieval": {
                "total_searched": len(candidates),
                "returned":       len(results),
                "results":        results,
            },
            "images": {
                "original": original_b64,
                "heatmap":  heatmap_b64,
            },
        }


# ============================================================
# FLASK APP
# FIX: static_folder points to 'static' subfolder
# ============================================================
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

engine = None


def get_engine():
    global engine
    if engine is None:
        engine = RetrievalEngine()
    return engine


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
@app.route("/index.html")
def index():
    return send_from_directory(".", "index.html")

@app.route("/analyse.html")
def analyse():
    return send_from_directory(".", "analyse.html")

@app.route("/about.html")
def about():
    return send_from_directory(".", "about.html")

@app.route("/contact.html")
def contact():
    return send_from_directory(".", "contact.html")

# FIX: Added missing @app.route decorator for /api/health
@app.route("/api/health")
def health():
    try:
        eng = get_engine()
        return jsonify({
            "status":     "ok",
            "model":      "Phase2Model (ResNet-50)",
            "index_size": eng.index.ntotal,
            "device":     str(eng.device),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
# ── Run ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  NeuroInsight — Flask Backend")
    print(f"  BASE_DIR: {BASE_DIR}")
    print("  Starting at http://localhost:8002")
    print("=" * 60)
    port = int(os.environ.get("PORT", 8002))
        app.run(debug=False, host="0.0.0.0", port=port)