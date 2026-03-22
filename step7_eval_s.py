"""
PHASE II - STEP 7: Complete Evaluation
=======================================
Full evaluation suite including:

SECTION 1 — CLASSIFICATION METRICS (per attribute)
  - Accuracy, Precision, Recall, F1 (macro + per class)
  - Confusion matrices for grade, severity, size, location

SECTION 2 — RETRIEVAL METRICS
  - Precision@K, Recall@K, NDCG@K, mAP
  - Per-class breakdown

SECTION 3 — COMPARISON TABLE
  - Training accuracy (from training logs) vs
  - Classification accuracy (this eval) vs
  - Retrieval precision (P@10)

OUTPUTS:
  - evaluation_output_enhanced/
      confusion_matrices.png
      metrics_chart.png
      classification_report.csv
      retrieval_metrics.csv
      evaluation_report.html   ← full report, open in browser
      metrics_summary.json
"""

import os
import sys
import json
import base64
import numpy as np
import pandas as pd
import faiss
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Add base path for model import
sys.path.append('/Users/sowmyaalamuri/Desktop/Capstone_project/Phase2')
from step2_model_s import Phase2Model

# ============================================================
# CONFIG
# ============================================================
BASE_DIR       = "/Users/sowmyaalamuri/Desktop/Capstone_project/Phase2"

FAISS_DIR      = os.path.join(BASE_DIR, "faiss_index_enhanced")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings_output_enhanced")
MODEL_PATH     = os.path.join(BASE_DIR, "training_output_enhanced", "best_model_s.pth")
OUTPUT_DIR     = os.path.join(BASE_DIR, "evaluation_output_enhanced")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training log from enhanced run
TRAINING_LOG   = os.path.join(BASE_DIR, "training_output_enhanced", "training_history_s.json")

EMBED_DIM      = 128
IMG_SIZE       = 160          # Match enhanced training size
N_QUERIES      = 200
K_VALUES       = [1, 5, 10, 20]
RANDOM_SEED    = 42

GRADE_MAP    = {0: "LGG",   1: "HGG"}
SEVERITY_MAP = {0: "low",   1: "medium", 2: "high"}
SIZE_MAP     = {0: "small", 1: "medium", 2: "large"}
LOCATION_MAP = {0: "left",  1: "right",  2: "bilateral"}

ATTR_CONFIG = {
    "grade":    {"map": GRADE_MAP,    "labels": ["LGG", "HGG"],
                 "true_col": "true_grade",    "pred_col": "pred_grade"},
    "severity": {"map": SEVERITY_MAP, "labels": ["low", "medium", "high"],
                 "true_col": "true_severity", "pred_col": "pred_severity"},
    "size":     {"map": SIZE_MAP,     "labels": ["small", "medium", "large"],
                 "true_col": "true_size",     "pred_col": "pred_size"},
    "location": {"map": LOCATION_MAP, "labels": ["left", "right", "bilateral"],
                 "true_col": "true_location", "pred_col": "pred_location"},
}


# ============================================================
# DEVICE
# ============================================================
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple MPS")
        torch.set_default_dtype(torch.float32)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using CUDA")
    else:
        device = torch.device("cpu")
        print("⚠ Using CPU")
    return device


# ============================================================
# LOAD RESOURCES
# ============================================================
def load_resources():
    print("[1/5] Loading resources...")
    device = get_device()

    # Load model
    print(f"  Loading model from: {os.path.basename(MODEL_PATH)}")
    model = Phase2Model(embedding_dim=EMBED_DIM, pretrained=False).to(device)

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  ✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            else:
                model.load_state_dict(checkpoint)
                print("  ✓ Model loaded successfully!")
        else:
            model.load_state_dict(checkpoint)
            print("  ✓ Model loaded successfully!")
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        return None, None, None, None, None, None

    model.eval()
    print(f"  ✓ Model ready (device={device})")

    # Load FAISS indexes
    flat_index_path  = os.path.join(FAISS_DIR, "faiss_flat.index")
    ivfpq_index_path = os.path.join(FAISS_DIR, "faiss_ivfpq.index")

    if not os.path.exists(flat_index_path):
        print(f"  ⚠ Flat index not found: {flat_index_path}")
        flat_index = None
    else:
        flat_index = faiss.read_index(flat_index_path)
        print(f"  ✓ Flat index loaded ({flat_index.ntotal} vectors)")

    if not os.path.exists(ivfpq_index_path):
        print(f"  ⚠ IVF-PQ index not found: {ivfpq_index_path}")
        ivfpq_index = None
    else:
        ivfpq_index = faiss.read_index(ivfpq_index_path)
        print(f"  ✓ IVF-PQ index loaded ({ivfpq_index.ntotal} vectors)")

    # Load embeddings
    embeddings_path = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    embeddings = np.load(embeddings_path).astype("float32")
    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms == 0, 1e-9, norms)

    # Load metadata
    metadata_path = os.path.join(EMBEDDINGS_DIR, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    print(f"  ✓ Embeddings {embeddings.shape}  Metadata {metadata.shape}")

    return device, model, flat_index, ivfpq_index, embeddings, metadata


# ============================================================
# SECTION 1 — CLASSIFICATION METRICS
# ============================================================
def compute_classification_metrics(metadata: pd.DataFrame) -> dict:
    """
    Compare true labels vs predicted labels from Step 4 metadata.
    Computes accuracy, precision, recall, F1 for each attribute.
    """
    print("\n[2/5] Computing classification metrics...")
    results = {}

    for attr, cfg in ATTR_CONFIG.items():
        true_col = cfg["true_col"]
        pred_col = cfg["pred_col"]

        # Fall back to alternate column names if needed
        if true_col not in metadata.columns:
            alt_true = f"true_{attr}_code"
            if alt_true in metadata.columns:
                true_col = alt_true

        if pred_col not in metadata.columns:
            alt_pred = f"pred_{attr}_code"
            if alt_pred in metadata.columns:
                pred_col = alt_pred

        if true_col not in metadata.columns or pred_col not in metadata.columns:
            print(f"  ⚠ Skipping {attr} — columns not found")
            continue

        y_true = metadata[true_col].tolist()
        y_pred = metadata[pred_col].tolist()
        labels = cfg["labels"]

        # Remove NaN / None values — works for both string and numeric columns
        valid  = [(t, p) for t, p in zip(y_true, y_pred)
                  if t is not None and p is not None
                  and not (isinstance(t, float) and np.isnan(t))
                  and not (isinstance(p, float) and np.isnan(p))]

        if len(valid) == 0:
            print(f"  ⚠ Skipping {attr} — no valid samples")
            continue

        y_true, y_pred = zip(*valid)

        # Detect whether columns are string labels or integer codes
        is_numeric = all(isinstance(v, (int, float)) for v in y_true)

        if is_numeric:
            # Integer-coded columns → convert to string labels for consistency
            label_map = cfg["map"]
            y_true = [label_map.get(int(v), str(int(v))) for v in y_true]
            y_pred = [label_map.get(int(v), str(int(v))) for v in y_pred]

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, labels=labels,
                               average="macro", zero_division=0)
        rec  = recall_score(y_true, y_pred, labels=labels,
                            average="macro", zero_division=0)
        f1   = f1_score(y_true, y_pred, labels=labels,
                        average="macro", zero_division=0)
        cm   = confusion_matrix(y_true, y_pred, labels=labels)
        cr   = classification_report(y_true, y_pred, labels=labels,
                                     output_dict=True, zero_division=0)

        results[attr] = {
            "accuracy":        round(acc, 4),
            "precision":       round(prec, 4),
            "recall":          round(rec, 4),
            "f1":              round(f1, 4),
            "confusion_matrix": cm,
            "labels":          labels,
            "per_class":       cr,
        }

        print(f"  ✓ {attr:<10}  acc={acc:.4f}  prec={prec:.4f}  "
              f"rec={rec:.4f}  f1={f1:.4f}")

    return results


# ============================================================
# PLOT CONFUSION MATRICES
# ============================================================
def plot_confusion_matrices(clf_results: dict, save_path: str):
    n_attrs = len(clf_results)
    fig, axes = plt.subplots(1, n_attrs, figsize=(5 * n_attrs, 5))
    fig.patch.set_facecolor("#0d0d0d")

    if n_attrs == 1:
        axes = [axes]

    attr_titles = {
        "grade":    "Grade (LGG / HGG)",
        "severity": "Severity",
        "size":     "Tumor Size",
        "location": "Location"
    }

    for ax, (attr, res) in zip(axes, clf_results.items()):
        cm     = res["confusion_matrix"]
        labels = res["labels"]

        # Normalize
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        sns.heatmap(
            cm_norm, ax=ax,
            annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor="#1a1a1a",
            cbar_kws={"shrink": 0.8}
        )

        # Raw counts overlay
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j + 0.5, i + 0.72,
                        f"n={cm[i,j]}",
                        ha="center", va="center",
                        fontsize=7, color="#aaa")

        ax.set_title(attr_titles.get(attr, attr),
                     color="white", fontsize=12, pad=10)
        ax.set_xlabel("Predicted", color="#aaa", fontsize=10)
        ax.set_ylabel("True",      color="#aaa", fontsize=10)
        ax.tick_params(colors="#ccc", labelsize=9)
        ax.set_facecolor("#0d0d0d")

        ax.text(0.98, 0.02, f"Acc={res['accuracy']:.3f}",
                transform=ax.transAxes, color="#4CAF50",
                fontsize=9, ha="right", fontweight="bold")

    plt.suptitle("Confusion Matrices — All Attributes",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0d0d0d")
    plt.close()
    print(f"  ✓ Confusion matrices saved: {save_path}")


# ============================================================
# SECTION 2 — RETRIEVAL METRICS
# ============================================================
# RELEVANCE RULES (used only for judging retrieved results,
# never for ranking/scoring):
#
#   Binary relevant  → same grade AND severity within ±1 step
#   Graded relevance → weighted score across all 4 attributes
#     grade    0.50  (exact match only)
#     severity 0.25 exact / 0.125 adjacent / 0.0 far
#     size     0.15  (exact match only)
#     location 0.10  (exact match only)
#
# RETRIEVAL: pure cosine similarity on L2-normalised embeddings.
# Ground truth labels are NEVER used for ranking — only for
# judging whether a returned item is relevant.
# ============================================================

def is_relevant(q_row, c_row) -> bool:
    """
    Binary relevance: same grade AND severity within ±1 step.
    Used for P@K, R@K, mAP flags.
    """
    if q_row["true_grade"] != c_row["true_grade"]:
        return False
    sev_order = ["low", "medium", "high"]
    q_s, c_s  = q_row["true_severity"], c_row["true_severity"]
    if q_s not in sev_order or c_s not in sev_order:
        return False
    return abs(sev_order.index(q_s) - sev_order.index(c_s)) <= 1


def graded_relevance(q_row, c_row) -> float:
    """
    Graded relevance score in [0, 1] across all 4 attributes.
    Used for NDCG (rewards partial matches).
    """
    score = 0.0
    if q_row["true_grade"] == c_row["true_grade"]:
        score += 0.50
    sev_order = ["low", "medium", "high"]
    q_s, c_s  = q_row["true_severity"], c_row["true_severity"]
    if q_s in sev_order and c_s in sev_order:
        d = abs(sev_order.index(q_s) - sev_order.index(c_s))
        score += 0.25 if d == 0 else 0.125 if d == 1 else 0.0
    if q_row["true_size"]     == c_row["true_size"]:     score += 0.15
    if q_row["true_location"] == c_row["true_location"]: score += 0.10
    return round(score, 4)


# ── Metric helpers ───────────────────────────────────────────

def precision_at_k(flags, k):
    return sum(flags[:k]) / k if k > 0 else 0.0

def recall_at_k(flags, k, tot):
    return sum(flags[:k]) / tot if tot > 0 else 0.0

def ndcg_at_k(scores, k):
    def dcg(s, k): return sum(v / np.log2(i + 2) for i, v in enumerate(s[:k]))
    ideal = dcg(sorted(scores, reverse=True), k)
    return dcg(scores, k) / ideal if ideal > 0 else 0.0

def average_precision(flags):
    hits, tot, ap = 0, 0, 0.0
    for i, r in enumerate(flags, 1):
        if r:
            hits += 1
            ap   += hits / i
            tot  += 1
    return ap / tot if tot > 0 else 0.0


# ── Precompute total relevant per query (full dataset scan) ──

def build_total_relevant_cache(metadata: pd.DataFrame) -> np.ndarray:
    """
    For every sample, count how many OTHER samples in the full
    dataset are binary-relevant to it.
    This is O(N²) but only runs once; with N=15k it takes ~10s.
    """
    print("  Pre-computing total_relevant for all queries "
          "(full dataset scan — done once)...")
    n = len(metadata)
    grades    = metadata["true_grade"].values
    severities= metadata["true_severity"].values
    sev_order = ["low", "medium", "high"]

    sev_idx = np.array([sev_order.index(s) if s in sev_order else -1
                        for s in severities])

    totals = np.zeros(n, dtype=np.int32)
    for i in range(n):
        same_grade = grades == grades[i]
        sev_dist   = np.abs(sev_idx - sev_idx[i])
        valid_sev  = (sev_idx[i] >= 0) & (sev_idx >= 0)
        relevant   = same_grade & valid_sev & (sev_dist <= 1)
        relevant[i] = False          # exclude self
        totals[i]   = int(relevant.sum())

    totals = np.maximum(totals, 1)   # avoid division by zero
    print(f"  ✓ total_relevant computed  "
          f"(mean={totals.mean():.1f}, min={totals.min()}, max={totals.max()})")
    return totals


# ── Per-query evaluation ─────────────────────────────────────

def evaluate_query(idx, embeddings, metadata, flat_index,
                   total_relevant_cache, max_k=20):
    """
    Retrieve top-(max_k) neighbours using COSINE SIMILARITY ONLY
    (pure embedding — no ground truth used for ranking).
    Judge relevance using ground truth labels AFTER retrieval.
    """
    q_row = metadata.iloc[idx]

    # Retrieve max_k+1 so we can drop self if it appears
    D, I = flat_index.search(embeddings[[idx]], max_k + 1)
    D, I = D[0], I[0]

    # Keep only valid non-self results, ranked by cosine similarity
    # Embeddings are L2-normalised → cosine = 1 - (L2²/2)
    ranked = []
    for dist, cidx in zip(D, I):
        if cidx == idx or cidx < 0:
            continue
        cosine = float(max(0.0, 1.0 - dist / 2.0))
        ranked.append((cosine, cidx))

    ranked = ranked[:max_k]   # top-K by embedding similarity only

    # Judge relevance using ground truth (correct — done AFTER retrieval)
    flags  = [1 if is_relevant(q_row, metadata.iloc[ci]) else 0
              for _, ci in ranked]
    graded = [graded_relevance(q_row, metadata.iloc[ci])
              for _, ci in ranked]

    # True total relevant from full-dataset cache (not just top-200)
    tot_rel = int(total_relevant_cache[idx])

    return {
        "query_idx":      idx,
        "grade":          q_row["true_grade"],
        "severity":       q_row["true_severity"],
        "relevant_flags": flags,
        "graded_scores":  graded,
        "total_relevant": tot_rel,
        "ap":             average_precision(flags),
        # store top cosine scores for diagnostics
        "top_cosines":    [round(c, 4) for c, _ in ranked[:5]],
    }


def compute_retrieval_metrics(results, k_values):
    m = defaultdict(list)
    for r in results:
        for k in k_values:
            m[f"P@{k}"].append(precision_at_k(r["relevant_flags"], k))
            m[f"R@{k}"].append(recall_at_k(r["relevant_flags"], k,
                                            r["total_relevant"]))
            m[f"NDCG@{k}"].append(ndcg_at_k(r["graded_scores"], k))
        m["AP"].append(r["ap"])
    summary = {k: round(float(np.mean(v)), 4) for k, v in m.items()}
    summary["mAP"] = summary.pop("AP")
    return summary


def compute_per_class_retrieval(results, k_values):
    classes = defaultdict(list)
    for r in results:
        classes[f"{r['grade']} | {r['severity']}"].append(r)
    rows = []
    for cls, cr in sorted(classes.items()):
        m = compute_retrieval_metrics(cr, k_values)
        rows.append({"class": cls, "n_queries": len(cr), **m})
    return pd.DataFrame(rows)


# ============================================================
# SECTION 3 — COMPARISON TABLE
# ============================================================
def load_training_results() -> dict:
    """Load training accuracy from log if available."""
    defaults = {
        "grade_acc":    "N/A",
        "severity_acc": "N/A",
        "size_acc":     "N/A",
        "location_acc": "N/A",
    }
    if os.path.exists(TRAINING_LOG):
        try:
            with open(TRAINING_LOG) as f:
                log = json.load(f)

            best_val = {}
            if isinstance(log, list) and len(log) > 0:
                best_val['grade_acc']    = max([e.get('val', {}).get('grade', 0)
                                                for e in log if 'val' in e])
                best_val['severity_acc'] = max([e.get('val', {}).get('severity', 0)
                                                for e in log if 'val' in e])
                best_val['size_acc']     = max([e.get('val', {}).get('size', 0)
                                                for e in log if 'val' in e])
                best_val['location_acc'] = max([e.get('val', {}).get('location', 0)
                                                for e in log if 'val' in e])
                return {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in best_val.items()}
            else:
                # Flat dict fallback
                return {
                    "grade_acc":    log.get("best_grade_acc",
                                    log.get("grade_accuracy", "N/A")),
                    "severity_acc": log.get("best_severity_acc",
                                    log.get("severity_accuracy", "N/A")),
                    "size_acc":     log.get("best_size_acc",
                                    log.get("size_accuracy", "N/A")),
                    "location_acc": log.get("best_location_acc",
                                    log.get("location_accuracy", "N/A")),
                }
        except Exception as e:
            print(f"  ⚠ Could not load training log: {e}")
    return defaults


# ============================================================
# PLOT METRICS CHART
# ============================================================
def plot_metrics_chart(ret_summary, per_class_df, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0d0d0d")

    k_vals = [1, 5, 10, 20]

    # Plot 1: P@K, R@K, NDCG@K
    ax = axes[0]
    ax.set_facecolor("#1a1a1a")
    for label, key, color in [("Precision@K", "P",    "#2196F3"),
                               ("Recall@K",    "R",    "#4CAF50"),
                               ("NDCG@K",      "NDCG", "#FF9800")]:
        vals = [ret_summary[f"{key}@{k}"] for k in k_vals]
        ax.plot(k_vals, vals, marker="o", color=color,
                linewidth=2, markersize=7, label=label)
        for k, v in zip(k_vals, vals):
            ax.annotate(f"{v:.2f}", (k, v),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=color)
    ax.set_title("Retrieval Metrics vs K", color="white", fontsize=12, pad=10)
    ax.set_xlabel("K", color="#aaa"); ax.set_ylabel("Score", color="#aaa")
    ax.set_xticks(k_vals); ax.set_ylim(0, 1.15)
    ax.tick_params(colors="#aaa"); ax.spines[:].set_color("#333")
    ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=9)
    ax.text(0.98, 0.05, f"mAP={ret_summary['mAP']:.3f}",
            transform=ax.transAxes, color="#FF9800",
            fontsize=10, ha="right", fontweight="bold")

    # Plot 2: P@10 per class
    ax = axes[1]
    ax.set_facecolor("#1a1a1a")
    cls     = per_class_df["class"].tolist()
    p10     = per_class_df["P@10"].tolist()
    bcolors = ["#9C27B0" if "HGG" in c else "#2196F3" for c in cls]
    bars    = ax.barh(cls, p10, color=bcolors, height=0.5)
    for bar, val in zip(bars, p10):
        ax.text(min(val + 0.02, 1.05), bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", color="white", fontsize=9)
    ax.set_title("Precision@10 per Class", color="white", fontsize=12, pad=10)
    ax.set_xlabel("Precision@10", color="#aaa"); ax.set_xlim(0, 1.2)
    ax.tick_params(colors="#aaa"); ax.spines[:].set_color("#333")

    # Plot 3: mAP per class
    ax = axes[2]
    ax.set_facecolor("#1a1a1a")
    maps = per_class_df["mAP"].tolist()
    bars = ax.barh(cls, maps, color=bcolors, height=0.5)
    for bar, val in zip(bars, maps):
        ax.text(min(val + 0.02, 1.05), bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", color="white", fontsize=9)
    ax.set_title("mAP per Class", color="white", fontsize=12, pad=10)
    ax.set_xlabel("mAP", color="#aaa"); ax.set_xlim(0, 1.2)
    ax.tick_params(colors="#aaa"); ax.spines[:].set_color("#333")
    hgg = mpatches.Patch(color="#9C27B0", label="HGG")
    lgg = mpatches.Patch(color="#2196F3", label="LGG")
    ax.legend(handles=[hgg, lgg], facecolor="#1a1a1a",
              labelcolor="white", fontsize=9)

    plt.suptitle("Retrieval Evaluation", color="white",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print(f"  ✓ Metrics chart saved: {save_path}")


# ============================================================
# HTML REPORT
# ============================================================
def img_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def color_cell(val, g=0.7, o=0.5):
    if isinstance(val, str):
        return f'<td style="text-align:center;color:#aaa">{val}</td>'
    c = "#4CAF50" if val >= g else "#FF9800" if val >= o else "#F44336"
    return f'<td style="text-align:center;color:{c};font-weight:bold">{val:.4f}</td>'


def generate_full_html(clf_results, ret_summary, per_class_df, ret_results,
                       train_results, cm_path, chart_path,
                       n_queries, save_path):

    cm_b64    = img_to_b64(cm_path)
    chart_b64 = img_to_b64(chart_path)

    # ---- Classification table ----
    clf_rows = ""
    for attr, res in clf_results.items():
        clf_rows += f"""
        <tr>
          <td style="color:#FF9800;font-weight:bold;text-transform:capitalize">{attr}</td>
          {color_cell(res['accuracy'])}
          {color_cell(res['precision'])}
          {color_cell(res['recall'])}
          {color_cell(res['f1'])}
        </tr>"""

    # ---- Per-class classification ----
    per_clf_rows = ""
    for attr, res in clf_results.items():
        for label in res["labels"]:
            if label not in res["per_class"]: continue
            pc = res["per_class"][label]
            per_clf_rows += f"""
            <tr>
              <td style="color:#FF9800;text-transform:capitalize">{attr}</td>
              <td style="color:#aaa">{label}</td>
              {color_cell(pc.get('precision', 0))}
              {color_cell(pc.get('recall', 0))}
              {color_cell(pc.get('f1-score', 0))}
              <td style="text-align:center;color:#666">{int(pc.get('support', 0))}</td>
            </tr>"""

    # ---- Retrieval table ----
    ret_rows = ""
    for k in K_VALUES:
        p = ret_summary[f"P@{k}"]
        r = ret_summary[f"R@{k}"]
        n = ret_summary[f"NDCG@{k}"]
        ret_rows += f"""
        <tr>
          <td style="color:#aaa;text-align:center">@{k}</td>
          {color_cell(p)} {color_cell(r)} {color_cell(n)}
        </tr>"""

    # ---- Per class retrieval ----
    pcls_rows = ""
    for _, row in per_class_df.iterrows():
        gc = "#9C27B0" if "HGG" in row["class"] else "#2196F3"
        pcls_rows += f"""
        <tr>
          <td style="color:{gc};font-weight:bold">{row['class']}</td>
          <td style="text-align:center;color:#666">{row['n_queries']}</td>
          {color_cell(row['P@5'])} {color_cell(row['P@10'])}
          {color_cell(row['R@10'])} {color_cell(row['NDCG@10'])}
          {color_cell(row['mAP'])}
        </tr>"""

    # ---- Comparison table ----
    # Build per-attribute P@10 by grouping retrieval results on grade and severity
    # (which are stored per query). For size and location, use the overall P@10.
    def _attr_p10(results, attr):
        groups = defaultdict(list)
        for r in results:
            groups[r[attr]].append(r)
        scores = [precision_at_k(r["relevant_flags"], 10)
                  for grp in groups.values() for r in grp]
        return round(float(np.mean(scores)), 4) if scores else "N/A"

    overall_p10 = round(float(ret_summary["P@10"]), 4)
    attr_ret_p10 = {
        "grade":    _attr_p10(ret_results, "grade"),
        "severity": _attr_p10(ret_results, "severity"),
        "size":     overall_p10,   # not tracked per-query; use global P@10
        "location": overall_p10,
    }

    attrs = ["grade", "severity", "size", "location"]

    def fmt(v):
        if isinstance(v, float): return f"{v:.4f}"
        return str(v)

    cmp_rows = ""
    for attr in attrs:
        train_acc = train_results.get(f"{attr}_acc", "N/A")
        eval_acc  = clf_results[attr]["accuracy"] if attr in clf_results else "N/A"
        eval_f1   = clf_results[attr]["f1"]       if attr in clf_results else "N/A"
        ret_p10   = attr_ret_p10.get(attr, "N/A")

        def fmt_cell(v, is_metric=True):
            if v == "N/A":
                return f'<td style="text-align:center;color:#555">{v}</td>'
            if is_metric:
                return color_cell(float(v))
            return f'<td style="text-align:center;color:#aaa">{fmt(v)}</td>'

        cmp_rows += f"""
        <tr>
          <td style="color:#FF9800;font-weight:bold;text-transform:capitalize">{attr}</td>
          {fmt_cell(train_acc)}
          {fmt_cell(eval_acc)}
          {fmt_cell(eval_f1)}
          {fmt_cell(ret_p10)}
        </tr>"""

    n_clf_samples = len(
        next(iter(clf_results.values()))["confusion_matrix"].flatten()
    ) if clf_results else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Complete Evaluation — Brain Tumor MRI Retrieval</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0 }}
    body {{
      background:#0a0a0a; color:#e0e0e0;
      font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
      padding:36px; max-width:1300px; margin:0 auto;
    }}
    h1  {{ font-size:22px; font-weight:700; margin-bottom:6px }}
    h2  {{ font-size:13px; font-weight:600; color:#666; margin:30px 0 12px;
           text-transform:uppercase; letter-spacing:1.5px;
           border-bottom:1px solid #222; padding-bottom:8px }}
    .card {{
      background:#111; border:1px solid #222;
      border-radius:12px; padding:20px; margin-bottom:20px
    }}
    table {{ width:100%; border-collapse:collapse; font-size:13px }}
    th {{ background:#1a1a1a; color:#555; padding:10px 14px;
          text-align:left; font-weight:600;
          border-bottom:1px solid #2a2a2a; font-size:11px;
          text-transform:uppercase; letter-spacing:0.5px }}
    td {{ padding:9px 14px; border-bottom:1px solid #161616 }}
    tr:last-child td {{ border-bottom:none }}
    tr:hover td {{ background:#141414 }}
    .big-metric {{
      display:inline-block; background:#161616; border:1px solid #2a2a2a;
      border-radius:10px; padding:16px 22px; margin:6px; text-align:center;
      min-width:100px
    }}
    .big-metric .val {{ font-size:26px; font-weight:700; color:#4CAF50 }}
    .big-metric .lbl {{
      font-size:11px; color:#555; margin-top:4px; text-transform:uppercase;
      letter-spacing:0.5px
    }}
    .legend {{ font-size:11px; color:#444; margin-top:12px }}
  </style>
</head>
<body>

  <h1>🧠 Brain Tumor MRI — Complete Evaluation Report</h1>
  <p style="font-size:12px;color:#444;margin-top:6px;margin-bottom:24px">
    Phase II · Step 7 · {n_queries} retrieval queries · {n_clf_samples} classification samples
  </p>

  <h2>Section 1 — Classification Metrics</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Attribute</th>
          <th style="text-align:center">Accuracy</th>
          <th style="text-align:center">Precision (macro)</th>
          <th style="text-align:center">Recall (macro)</th>
          <th style="text-align:center">F1 (macro)</th>
        </tr>
      </thead>
      <tbody>{clf_rows}</tbody>
    </table>
  </div>

  <h2>Per-Class Classification Breakdown</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Attribute</th><th>Class</th>
          <th style="text-align:center">Precision</th>
          <th style="text-align:center">Recall</th>
          <th style="text-align:center">F1</th>
          <th style="text-align:center">Support</th>
        </tr>
      </thead>
      <tbody>{per_clf_rows}</tbody>
    </table>
  </div>

  <h2>Confusion Matrices</h2>
  <div class="card" style="text-align:center">
    <img src="data:image/png;base64,{cm_b64}"
         style="max-width:100%;border-radius:8px"/>
  </div>

  <h2>Section 2 — Retrieval Metrics</h2>
  <div class="card">
    <div class="big-metric">
      <div class="val">{ret_summary['mAP']:.3f}</div>
      <div class="lbl">mAP</div>
    </div>
    <div class="big-metric">
      <div class="val">{ret_summary['P@10']:.3f}</div>
      <div class="lbl">Precision@10</div>
    </div>
    <div class="big-metric">
      <div class="val">{ret_summary['R@10']:.3f}</div>
      <div class="lbl">Recall@10</div>
    </div>
    <div class="big-metric">
      <div class="val">{ret_summary['NDCG@10']:.3f}</div>
      <div class="lbl">NDCG@10</div>
    </div>
    <div class="big-metric">
      <div class="val">{ret_summary['P@5']:.3f}</div>
      <div class="lbl">Precision@5</div>
    </div>
  </div>

  <div class="card">
    <table>
      <thead>
        <tr>
          <th>K</th>
          <th style="text-align:center">Precision@K</th>
          <th style="text-align:center">Recall@K</th>
          <th style="text-align:center">NDCG@K</th>
        </tr>
      </thead>
      <tbody>{ret_rows}</tbody>
    </table>
    <div style="margin-top:14px;font-size:13px">
      <span style="color:#FF9800;font-weight:bold">mAP = {ret_summary['mAP']:.4f}</span>
    </div>
  </div>

  <h2>Retrieval Per-Class Breakdown</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Class</th><th style="text-align:center">Queries</th>
          <th style="text-align:center">P@5</th>
          <th style="text-align:center">P@10</th>
          <th style="text-align:center">R@10</th>
          <th style="text-align:center">NDCG@10</th>
          <th style="text-align:center">mAP</th>
        </tr>
      </thead>
      <tbody>{pcls_rows}</tbody>
    </table>
  </div>

  <h2>Retrieval Charts</h2>
  <div class="card" style="text-align:center">
    <img src="data:image/png;base64,{chart_b64}"
         style="max-width:100%;border-radius:8px"/>
  </div>

  <h2>Section 3 — Training vs Evaluation vs Retrieval Comparison</h2>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Attribute</th>
          <th style="text-align:center">Training Accuracy</th>
          <th style="text-align:center">Eval Accuracy</th>
          <th style="text-align:center">Eval F1 (macro)</th>
          <th style="text-align:center">Retrieval P@10</th>
        </tr>
      </thead>
      <tbody>{cmp_rows}</tbody>
    </table>
    <p style="font-size:11px;color:#444;margin-top:12px">
      * Training Accuracy: from training log (if available) ·
      Retrieval P@10 shown for grade only (primary attribute)
    </p>
  </div>

  <div class="legend">
    <span style="color:#4CAF50">■</span> ≥ 0.70 &nbsp;
    <span style="color:#FF9800">■</span> 0.50–0.70 &nbsp;
    <span style="color:#F44336">■</span> &lt; 0.50
  </div>

  <div style="font-size:11px;color:#2a2a2a;margin-top:24px;text-align:center">
    Phase II · Step 7 · Brain Tumor MRI Retrieval System
  </div>
</body>
</html>"""

    with open(save_path, "w") as f:
        f.write(html)
    print(f"  ✓ Full HTML report saved: {save_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*70)
    print("  PHASE II - STEP 7: Complete Evaluation")
    print("="*70)

    device, model, flat_index, ivfpq_index, embeddings, metadata = load_resources()

    if flat_index is None:
        print("❌ Cannot proceed without FAISS flat index.")
        return

    # ── SECTION 1: Classification ──────────────────────────
    clf_results = compute_classification_metrics(metadata)

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plot_confusion_matrices(clf_results, cm_path)

    # ── SECTION 2: Retrieval ───────────────────────────────
    # Pre-compute total_relevant for every sample from the full
    # dataset — this ensures recall denominators are correct.
    print(f"\n[3/5] Sampling {N_QUERIES} queries (stratified) + "
          f"building relevance cache...")
    total_relevant_cache = build_total_relevant_cache(metadata)

    np.random.seed(RANDOM_SEED)
    query_indices = []
    per_class_n   = N_QUERIES // 6
    for grade in ["HGG", "LGG"]:
        for sev in ["low", "medium", "high"]:
            subset = metadata[
                (metadata["true_grade"] == grade) &
                (metadata["true_severity"] == sev)
            ].index.tolist()
            n = min(per_class_n, len(subset))
            query_indices.extend(
                np.random.choice(subset, n, replace=False).tolist())
            print(f"  ✓ {grade} | {sev:<6}: {n} queries")

    print(f"\n[4/5] Evaluating {len(query_indices)} retrieval queries "
          f"(pure cosine similarity — no ground truth in ranking)...")
    ret_results = []
    for idx in tqdm(query_indices, desc="  Retrieval eval"):
        ret_results.append(
            evaluate_query(idx, embeddings, metadata, flat_index,
                           total_relevant_cache, max_k=max(K_VALUES)))

    ret_summary  = compute_retrieval_metrics(ret_results, K_VALUES)
    per_class_df = compute_per_class_retrieval(ret_results, K_VALUES)

    # ── SECTION 3: Comparison ──────────────────────────────
    train_results = load_training_results()

    # ── Charts ────────────────────────────────────────────
    print(f"\n[5/5] Generating outputs...")
    chart_path = os.path.join(OUTPUT_DIR, "metrics_chart.png")
    plot_metrics_chart(ret_summary, per_class_df, chart_path)

    # ── HTML Report ───────────────────────────────────────
    html_path = os.path.join(OUTPUT_DIR, "evaluation_report.html")
    generate_full_html(
        clf_results, ret_summary, per_class_df, ret_results,
        train_results, cm_path, chart_path,
        len(query_indices), html_path
    )

    # ── Save CSVs + JSON ──────────────────────────────────
    clf_df = pd.DataFrame([
        {"attribute": a, **{k: v for k, v in r.items()
         if k not in ["confusion_matrix", "per_class", "labels"]}}
        for a, r in clf_results.items()
    ])
    clf_df.to_csv(os.path.join(OUTPUT_DIR, "classification_report.csv"), index=False)
    per_class_df.to_csv(os.path.join(OUTPUT_DIR, "retrieval_metrics.csv"), index=False)

    with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump({
            "classification": {
                a: {k: v for k, v in r.items()
                    if k not in ["confusion_matrix", "per_class", "labels"]}
                for a, r in clf_results.items()
            },
            "retrieval":              ret_summary,
            "n_retrieval_queries":    len(query_indices),
        }, f, indent=2)

    # ── Print summary ─────────────────────────────────────
    print("\n" + "="*70)
    print("  CLASSIFICATION SUMMARY")
    print("="*70)
    print(f"  {'Attribute':<12} {'Accuracy':>9} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>8}")
    print(f"  {'-'*50}")
    for attr, res in clf_results.items():
        print(f"  {attr:<12} {res['accuracy']:>9.4f} {res['precision']:>10.4f} "
              f"{res['recall']:>8.4f} {res['f1']:>8.4f}")

    print("\n" + "="*70)
    print("  RETRIEVAL SUMMARY")
    print("="*70)
    print(f"  {'':12} {'@1':>8} {'@5':>8} {'@10':>8} {'@20':>8}")
    print(f"  {'-'*44}")
    for m in ["P", "R", "NDCG"]:
        name = {"P": "Precision", "R": "Recall", "NDCG": "NDCG"}[m]
        vals = [ret_summary[f"{m}@{k}"] for k in K_VALUES]
        print(f"  {name:<12} " + "  ".join(f"{v:>6.4f}" for v in vals))
    print(f"\n  mAP: {ret_summary['mAP']:.4f}")

    print("\n" + "="*70)
    print("  ✓ STEP 7 COMPLETE!")
    print("="*70)
    print(f"""
  Outputs → {OUTPUT_DIR}
    confusion_matrices.png
    metrics_chart.png
    classification_report.csv
    retrieval_metrics.csv
    metrics_summary.json
    evaluation_report.html   ← open this!

  To open:
    open {html_path}

  Ready for Step 8 — FastAPI Backend! 🚀
{"="*70}
""")


if __name__ == "__main__":
    main()