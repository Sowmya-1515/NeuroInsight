"""
NeuroInsight — Medical-Grade PDF Report Generator
===================================================
Produces a hospital-style radiology report using ReportLab canvas.
Now uses real patient data passed from the frontend modal, and shows
a clean "No Tumor Detected" page when tumor_present is False.
"""

import io
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm

PAGE_W, PAGE_H = A4
ML = 20 * mm
MR = 20 * mm
MB = 18 * mm
CW = PAGE_W - ML - MR

C_BLACK   = colors.HexColor("#0d0d0d")
C_NAVY    = colors.HexColor("#0a2240")
C_NAVY_LT = colors.HexColor("#1a3a60")
C_RULE    = colors.HexColor("#b0bec5")
C_LIGHT   = colors.HexColor("#f0f4f8")
C_RED     = colors.HexColor("#c0392b")
C_ORANGE  = colors.HexColor("#d35400")
C_GREEN   = colors.HexColor("#1a6b3c")
C_BLUE    = colors.HexColor("#1565c0")
C_MUTED   = colors.HexColor("#546e7a")
C_WHITE   = colors.white
C_BORDER  = colors.HexColor("#cfd8dc")
C_SUCCESS = colors.HexColor("#1a6b3c")


def severity_color(sev):
    return {"high": C_RED, "medium": C_ORANGE, "low": C_GREEN}.get(sev, C_MUTED)

def grade_color(g):
    return C_RED if g == "HGG" else C_BLUE


# ── Helpers ────────────────────────────────────────────────────────────────

def measure_wrap_height(cv, text, max_w, font, size, line_h):
    """Return total height that wrap_text will consume, without drawing."""
    words = text.split()
    line  = ""
    lines = 0
    for word in words:
        test = (line + " " + word).strip()
        if cv.stringWidth(test, font, size) > max_w:
            lines += 1
            line = word
        else:
            line = test
    if line:
        lines += 1
    return lines * line_h


def wrap_text(cv, text, x, y, max_w, font="Helvetica", size=8.5,
              color=C_BLACK, line_h=5 * mm):
    cv.setFont(font, size)
    cv.setFillColor(color)
    words = text.split()
    line  = ""
    for word in words:
        test = (line + " " + word).strip()
        if cv.stringWidth(test, font, size) > max_w:
            cv.drawString(x, y, line)
            y   -= line_h
            line = word
        else:
            line = test
    if line:
        cv.drawString(x, y, line)
        y -= line_h
    return y


def mini_bar(cv, x, y, pct, bar_w=38 * mm, h=3 * mm, color=C_NAVY):
    cv.setFillColor(C_LIGHT)
    cv.rect(x, y, bar_w, h, fill=1, stroke=0)
    cv.setFillColor(color)
    cv.rect(x, y, bar_w * min(pct / 100.0, 1.0), h, fill=1, stroke=0)
    cv.setFont("Helvetica", 7)
    cv.setFillColor(C_MUTED)
    cv.drawString(x + bar_w + 2 * mm, y + 0.3 * mm, f"{pct:.0f}%")


def section_heading(cv, y, title):
    cv.setFillColor(C_NAVY)
    cv.rect(ML, y - 5.5 * mm, CW, 6.5 * mm, fill=1, stroke=0)
    cv.setFillColor(C_WHITE)
    cv.setFont("Helvetica-Bold", 8)
    cv.drawString(ML + 3 * mm, y - 2.8 * mm, title.upper())
    return y - 5.5 * mm - 3 * mm


def ruled_field(cv, y, label, value, bold_val=False, val_col=C_BLACK,
                col2_label=None, col2_val=None):
    cv.setFont("Helvetica", 8); cv.setFillColor(C_MUTED)
    cv.drawString(ML, y, label)
    cv.setFont("Helvetica-Bold" if bold_val else "Helvetica", 8.5)
    cv.setFillColor(val_col)
    cv.drawString(ML + 50 * mm, y, str(value))
    if col2_label:
        cv.setFont("Helvetica", 8); cv.setFillColor(C_MUTED)
        cv.drawString(ML + CW / 2 + 5 * mm, y, col2_label)
        cv.setFont("Helvetica", 8.5); cv.setFillColor(C_BLACK)
        cv.drawString(ML + CW / 2 + 55 * mm, y, str(col2_val))
    cv.setStrokeColor(C_BORDER); cv.setLineWidth(0.3)
    cv.setDash([1, 2], 0)
    cv.line(ML, y - 1.5 * mm, ML + CW, y - 1.5 * mm)
    cv.setDash([], 0)
    return y - 6.5 * mm


# ── Page frame ─────────────────────────────────────────────────────────────

def draw_page_frame(cv, page_num, total_pages, report_id):
    W, H = PAGE_W, PAGE_H
    cv.setFillColor(C_NAVY)
    cv.rect(0, H - 22 * mm, W, 22 * mm, fill=1, stroke=0)
    cv.setFillColor(colors.HexColor("#00acc1"))
    cv.rect(0, H - 22 * mm, W, 1.5 * mm, fill=1, stroke=0)
    cv.setFillColor(C_WHITE); cv.setFont("Helvetica-Bold", 16)
    cv.drawString(ML, H - 11 * mm, "NEUROINSIGHT")
    cv.setFont("Helvetica", 7.5); cv.setFillColor(colors.HexColor("#90b8d8"))
    cv.drawString(ML, H - 16.5 * mm,
        "AI-Assisted Neuroradiology Reporting System  |  ResNet-50 Phase II  |  FAISS IVF-PQ Retrieval")
    cv.setFillColor(C_WHITE); cv.setFont("Helvetica-Bold", 10)
    cv.drawRightString(W - MR, H - 11 * mm, "RADIOLOGY REPORT")
    cv.setFont("Helvetica", 7.5); cv.setFillColor(colors.HexColor("#90b8d8"))
    cv.drawRightString(W - MR, H - 16.5 * mm, f"Report ID: {report_id}")
    cv.setStrokeColor(C_RULE); cv.setLineWidth(0.5)
    cv.line(ML, MB + 8 * mm, W - MR, MB + 8 * mm)
    cv.setFont("Helvetica-Oblique", 6.5); cv.setFillColor(C_MUTED)
    cv.drawString(ML, MB + 4 * mm,
        "CONFIDENTIAL — FOR AUTHORISED MEDICAL PERSONNEL ONLY  "
        "|  AI-generated for research use only — not a substitute for clinical diagnosis")
    cv.drawRightString(W - MR, MB + 4 * mm, f"Page {page_num} of {total_pages}")
    cv.setStrokeColor(C_NAVY); cv.setLineWidth(1)
    cv.line(0, MB, W, MB)


# ── Patient info block (shared between both report types) ──────────────────

def draw_patient_block(cv, patient, report_id, now_str):
    """Draw the Patient & Study Information section. Returns new y."""
    y = PAGE_H - 22 * mm - 9 * mm
    y = section_heading(cv, y, "Patient & Study Information")

    cv.setFillColor(colors.HexColor("#f8fbff"))
    cv.setStrokeColor(C_BORDER); cv.setLineWidth(0.4)
    cv.rect(ML, y - 40 * mm, CW, 41 * mm, fill=1, stroke=1)

    iy = y - 2 * mm

    def info_row(l1, v1, l2, v2):
        nonlocal iy
        half = CW / 2 - 5 * mm
        cv.setFont("Helvetica", 7.5);    cv.setFillColor(C_MUTED)
        cv.drawString(ML + 2 * mm, iy, l1)
        cv.setFont("Helvetica-Bold", 8); cv.setFillColor(C_BLACK)
        cv.drawString(ML + 2 * mm + 40 * mm, iy, v1)
        cv.setFont("Helvetica", 7.5);    cv.setFillColor(C_MUTED)
        cv.drawString(ML + half + 7 * mm, iy, l2)
        cv.setFont("Helvetica-Bold", 8); cv.setFillColor(C_BLACK)
        cv.drawString(ML + half + 7 * mm + 40 * mm, iy, v2)
        cv.setStrokeColor(C_BORDER); cv.setLineWidth(0.25)
        cv.setDash([1, 3], 0)
        cv.line(ML + 2 * mm, iy - 1.5 * mm, ML + CW - 2 * mm, iy - 1.5 * mm)
        cv.setDash([], 0)
        iy -= 6 * mm

    info_row("Patient Name:",        patient.get("name", "[ANONYMISED]"),
             "Study Date:",          now_str[:11])
    info_row("Patient ID:",          patient.get("patientId", "NI-2026-XXXX"),
             "Report Date:",         now_str)
    info_row("Date of Birth:",       patient.get("dob", "__ / __ / ____"),
             "Accession No.:",       report_id)
    info_row("Referring Physician:", patient.get("physician", "[PHYSICIAN NAME]"),
             "Institution:",         "NeuroInsight Research")
    info_row("Modality:",            "MRI Brain",
             "Sequences:",           patient.get("sequences", "T1 / T2 / FLAIR Axial"))
    info_row("Clinical History:",    patient.get("history", "Suspected intracranial neoplasm"),
             "AI System:",           "NeuroInsight v1.0")

    return y - 42 * mm


# ── Page 1 — Tumor DETECTED ────────────────────────────────────────────────

def draw_page1_tumor(cv, result, patient, report_id, now_str):
    d    = result["diagnosis"]
    conf = result["confidence"]
    ret  = result["retrieval"]
    ms   = result["inference_ms"]
    gm   = sum(1 for r in ret["results"] if r["diagnosis"]["grade"] == d["grade"])
    avg_sim = sum(r["similarity"] for r in ret["results"]) / max(len(ret["results"]), 1)

    y = draw_patient_block(cv, patient, report_id, now_str)

    # ── AI Classification Findings ──────────────────────────────────────────
    y = section_heading(cv, y, "AI Classification Findings")

    box_h = 68 * mm
    cv.setFillColor(colors.HexColor("#f8fbff"))
    cv.setStrokeColor(C_BORDER); cv.setLineWidth(0.5)
    cv.rect(ML, y - box_h + 4 * mm, CW, box_h, fill=1, stroke=1)

    mid_x = ML + CW / 2
    mid_y = y - box_h / 2 + 4 * mm
    cv.setLineWidth(0.3); cv.setStrokeColor(C_BORDER)
    cv.line(mid_x, y + 1 * mm,    mid_x, y - box_h + 5 * mm)
    cv.line(ML + 2 * mm, mid_y,    ML + CW - 2 * mm, mid_y)

    findings = [
        ("TUMOR GRADE",    d["grade"],    conf["grade"],    grade_color(d["grade"]),       ML + 5 * mm,          y + 1 * mm),
        ("SEVERITY LEVEL", d["severity"], conf["severity"], severity_color(d["severity"]), ML + CW / 2 + 5 * mm, y + 1 * mm),
        ("TUMOR SIZE",     d["size"],     conf["size"],     C_NAVY,                        ML + 5 * mm,          mid_y + 1 * mm),
        ("LOCATION",       d["location"], conf["location"], C_NAVY,                        ML + CW / 2 + 5 * mm, mid_y + 1 * mm),
    ]
    for label, val, pct, col, fx, fy in findings:
        cv.setFont("Helvetica-Bold", 7); cv.setFillColor(C_MUTED)
        cv.drawString(fx, fy - 3 * mm, label)
        cv.setFont("Helvetica-Bold", 16); cv.setFillColor(col)
        cv.drawString(fx, fy - 11 * mm, val.upper())
        cv.setFont("Helvetica", 7.5); cv.setFillColor(C_MUTED)
        cv.drawString(fx, fy - 16 * mm, "Confidence:")
        mini_bar(cv, fx + 23 * mm, fy - 17 * mm, pct, bar_w=36 * mm, h=4 * mm, color=col)

    y -= box_h - 2 * mm

    # Severity alert
    sev  = d["severity"]
    scol = severity_color(sev)
    msg  = {
        "high":   "URGENT: High-grade features identified. Immediate specialist review strongly recommended.",
        "medium": "ADVISORY: Moderate-grade features present. Specialist review and follow-up recommended.",
        "low":    "ROUTINE: Low-grade features. Routine monitoring and follow-up advised.",
    }.get(sev, "")
    cv.setFillColor(colors.Color(scol.red, scol.green, scol.blue, alpha=0.07))
    cv.setStrokeColor(scol); cv.setLineWidth(1.2)
    cv.roundRect(ML, y - 9 * mm, CW, 9 * mm, 1.5 * mm, fill=1, stroke=1)
    prefix   = f"[{sev.upper()} SEVERITY]"
    prefix_w = cv.stringWidth(prefix, "Helvetica-Bold", 8.5)
    cv.setFont("Helvetica-Bold", 8.5); cv.setFillColor(scol)
    cv.drawString(ML + 3 * mm, y - 4.5 * mm, prefix)
    cv.setFont("Helvetica", 8.5); cv.setFillColor(C_BLACK)
    cv.drawString(ML + 3 * mm + prefix_w + 4, y - 4.5 * mm, msg)
    y -= 13 * mm

    # ── Technical Parameters ────────────────────────────────────────────────
    y = section_heading(cv, y, "Technical Parameters")
    y = ruled_field(cv, y, "AI Model:",                  "ResNet-50  —  Phase II Multi-Task Learning", bold_val=True)
    y = ruled_field(cv, y, "Embedding Space:",           "128-dimensional L2-normalised feature vectors")
    y = ruled_field(cv, y, "Retrieval Index:",           f"FAISS IVF-PQ  —  {ret['total_searched']} candidate cases searched")
    y = ruled_field(cv, y, "Ranking Formula:",           "60% cosine embedding similarity  +  40% weighted attribute score")
    y = ruled_field(cv, y, "Attribute Weights:",         "Grade 35%  |  Severity 30%  |  Size 20%  |  Location 15%")
    y = ruled_field(cv, y, "Inference Time:",            f"{ms} ms",
                    col2_label="Grade Match Rate:", col2_val=f"{gm}/10  ({gm * 10}%)")
    y = ruled_field(cv, y, "Avg. Retrieval Similarity:", f"{avg_sim:.1f}%", bold_val=True)
    y -= 4 * mm

    # ── Impression ──────────────────────────────────────────────────────────
    y = section_heading(cv, y, "Impression")

    grade_long = "HIGH-GRADE GLIOMA (HGG)" if d["grade"] == "HGG" else "LOW-GRADE GLIOMA (LGG)"
    history    = patient.get("history", "Suspected intracranial neoplasm")
    imps = [
        f"MRI brain demonstrates features consistent with a {d['size'].upper()} "
        f"{grade_long} in the {d['location'].upper()} hemisphere. "
        f"Grade confidence: {conf['grade']:.0f}%.",

        f"Tumour severity classified as {d['severity'].upper()} "
        f"(confidence {conf['severity']:.0f}%). "
        f"Tumour size: {d['size'].upper()} (confidence {conf['size']:.0f}%). "
        f"Location: {d['location'].upper()} (confidence {conf['location']:.0f}%).",

        f"FAISS retrieval identified {gm} grade-matched cases among top 10 results "
        f"(average similarity {avg_sim:.1f}%). Correlate with clinical findings.",

        f"Clinical context: {history}",
    ]

    # Dynamically size the impression box
    imp_line_h  = 4.8 * mm
    imp_gap     = 1.5 * mm
    wrap_max_w  = CW - 12 * mm
    content_h   = sum(
        measure_wrap_height(cv, txt, wrap_max_w, "Helvetica", 8.5, imp_line_h) + imp_gap
        for txt in imps
    ) - imp_gap + 6 * mm   # top+bottom padding
    imp_h = max(content_h, 28 * mm)

    cv.setFillColor(colors.HexColor("#f8fbff"))
    cv.setStrokeColor(C_BORDER); cv.setLineWidth(0.5)
    cv.rect(ML, y - imp_h + 4 * mm, CW, imp_h, fill=1, stroke=1)

    iy2 = y - 2 * mm
    for n, txt in enumerate(imps, 1):
        cv.setFont("Helvetica-Bold", 8.5); cv.setFillColor(C_NAVY)
        cv.drawString(ML + 3 * mm, iy2, f"{n}.")
        iy2 = wrap_text(cv, txt, ML + 9 * mm, iy2, wrap_max_w,
                        font="Helvetica", size=8.5, line_h=imp_line_h)
        iy2 -= imp_gap

    y -= imp_h - 2 * mm

# ── Page 1 — NO Tumor (clean summary) ──────────────────────────────────────

def draw_page1_no_tumor(cv, result, patient, report_id, now_str):
    """Single-page report for scans where no tumor is detected."""
    d   = result["diagnosis"]
    ms  = result["inference_ms"]
    conf_tumor = d.get("tumor_confidence", None)

    y = draw_patient_block(cv, patient, report_id, now_str)

    # ── Result banner ───────────────────────────────────────────────────────
    y = section_heading(cv, y, "AI Classification Result")

    banner_h = 38 * mm
    cv.setFillColor(colors.HexColor("#edfaf3"))
    cv.setStrokeColor(colors.HexColor("#6ee7b7")); cv.setLineWidth(1.5)
    cv.roundRect(ML, y - banner_h + 4 * mm, CW, banner_h, 3 * mm, fill=1, stroke=1)

    # Large checkmark icon
    icon_cx = ML + 22 * mm
    icon_cy = y - banner_h / 2 + 4 * mm
    cv.setFillColor(colors.HexColor("#d1fae5"))
    cv.circle(icon_cx, icon_cy, 13 * mm, fill=1, stroke=0)
    cv.setFont("Helvetica-Bold", 22); cv.setFillColor(C_SUCCESS)
    cv.drawCentredString(icon_cx, icon_cy - 4 * mm, "✓")

    # Text
    tx = ML + 40 * mm
    cv.setFont("Helvetica-Bold", 18); cv.setFillColor(C_SUCCESS)
    cv.drawString(tx, y - 10 * mm, "NO TUMOR DETECTED")
    cv.setFont("Helvetica", 9); cv.setFillColor(colors.HexColor("#065f46"))
    cv.drawString(tx, y - 17 * mm,
        "The AI model found no evidence of intracranial neoplasm in this MRI scan.")
    if conf_tumor is not None:
        cv.setFont("Helvetica-Bold", 8.5); cv.setFillColor(C_MUTED)
        cv.drawString(tx, y - 23 * mm, f"Detection confidence: {conf_tumor:.1f}%")
    cv.setFont("Helvetica-Oblique", 8); cv.setFillColor(C_MUTED)
    cv.drawString(tx, y - 29 * mm,
        "Routine monitoring advised. Correlate with clinical presentation.")

    y -= banner_h + 4 * mm

    # ── Technical Parameters ────────────────────────────────────────────────
    y = section_heading(cv, y, "Technical Parameters")
    y = ruled_field(cv, y, "AI Model:",         "ResNet-50  —  Phase II Multi-Task Learning", bold_val=True)
    y = ruled_field(cv, y, "Inference Time:",   f"{ms} ms",
                    col2_label="Tumor Detected:", col2_val="No")
    if conf_tumor is not None:
        y = ruled_field(cv, y, "Detection Confidence:", f"{conf_tumor:.1f}%", bold_val=True,
                        val_col=C_SUCCESS)
    y = ruled_field(cv, y, "Retrieval:",        "Skipped — no tumor features to index")
    y -= 4 * mm

    # ── Impression ──────────────────────────────────────────────────────────
    y = section_heading(cv, y, "Impression")

    history = patient.get("history", "—")
    imps = [
        "No intracranial neoplasm identified on the submitted MRI slice. "
        "The AI model did not detect features consistent with HGG or LGG glioma.",

        "Clinical correlation is strongly recommended. A single MRI slice may not be "
        "representative of the full examination. Follow-up imaging should be considered "
        "if clinical suspicion remains.",

        f"Clinical context: {history}",
    ]

    imp_line_h  = 4.8 * mm
    imp_gap     = 1.5 * mm
    wrap_max_w  = CW - 12 * mm
    content_h   = sum(
        measure_wrap_height(cv, txt, wrap_max_w, "Helvetica", 8.5, imp_line_h) + imp_gap
        for txt in imps
    ) - imp_gap + 6 * mm
    imp_h = max(content_h, 28 * mm)

    cv.setFillColor(colors.HexColor("#f8fbff"))
    cv.setStrokeColor(C_BORDER); cv.setLineWidth(0.5)
    cv.rect(ML, y - imp_h + 4 * mm, CW, imp_h, fill=1, stroke=1)

    iy2 = y - 2 * mm
    for n, txt in enumerate(imps, 1):
        cv.setFont("Helvetica-Bold", 8.5); cv.setFillColor(C_NAVY)
        cv.drawString(ML + 3 * mm, iy2, f"{n}.")
        iy2 = wrap_text(cv, txt, ML + 9 * mm, iy2, wrap_max_w,
                        font="Helvetica", size=8.5, line_h=imp_line_h)
        iy2 -= imp_gap

    y -= imp_h - 2 * mm

    # ── Disclaimer box ───────────────────────────────────────────────────────
    y -= 4 * mm
    cv.setFillColor(colors.HexColor("#fff7ed"))
    cv.setStrokeColor(C_ORANGE); cv.setLineWidth(0.8)
    cv.roundRect(ML, y - 16 * mm, CW, 16 * mm, 2 * mm, fill=1, stroke=1)
    cv.setFont("Helvetica-Bold", 8); cv.setFillColor(C_ORANGE)
    cv.drawString(ML + 3 * mm, y - 5 * mm, "⚠  DISCLAIMER")
    cv.setFont("Helvetica", 7.5); cv.setFillColor(C_BLACK)
    wrap_text(cv, "This AI result does not constitute a clinical diagnosis. "
              "A negative AI result should always be reviewed by a qualified radiologist "
              "before any clinical decision is made. NeuroInsight is a research tool only.",
              ML + 3 * mm, y - 10 * mm, CW - 6 * mm,
              font="Helvetica", size=7.5, line_h=4 * mm)
    y -= 20 * mm

# ── Page 2 — Retrieval (tumor detected only) ────────────────────────────────

def draw_page2(cv, result, report_id):
    W, H = PAGE_W, PAGE_H
    d    = result["diagnosis"]
    rows = result["retrieval"]["results"]
    y    = H - 22 * mm - 9 * mm

    y = section_heading(cv, y, "Similar Case Retrieval  —  FAISS IVF-PQ Index")
    cv.setFont("Helvetica", 8); cv.setFillColor(C_MUTED)
    cv.drawString(ML, y + 2 * mm,
        "Top-10 most similar MRI cases from the indexed database, "
        "ranked by combined embedding + attribute score.")
    y -= 9 * mm

    cols = [12*mm, 20*mm, 24*mm, 22*mm, 24*mm, 24*mm, 24*mm, 20*mm]
    hdrs = ["Rank","Grade","Severity","Size","Location","Similarity","Embed. Sim","Attr. Sim"]

    cv.setFillColor(C_NAVY)
    cv.rect(ML, y - 5.5*mm, CW, 6.5*mm, fill=1, stroke=0)
    cv.setFillColor(C_WHITE); cv.setFont("Helvetica-Bold", 7.5)
    hx = ML
    for w, h_lbl in zip(cols, hdrs):
        cv.drawCentredString(hx + w/2, y - 3*mm, h_lbl); hx += w
    y -= 5.5*mm

    for i, r in enumerate(rows):
        rd    = r["diagnosis"]
        match = rd["grade"] == d["grade"]
        row_bg = (colors.HexColor("#eaf7f0") if match else
                  colors.HexColor("#f8fbff") if i % 2 == 0 else C_WHITE)
        cv.setFillColor(row_bg)
        cv.rect(ML, y - 6*mm, CW, 6.5*mm, fill=1, stroke=0)
        cv.setStrokeColor(C_BORDER); cv.setLineWidth(0.3)
        cv.line(ML, y - 6*mm, ML + CW, y - 6*mm)
        rx = ML
        grade_txt = rd["grade"] + (" ✓" if match else "")
        vals = [
            (f"#{r['rank']}",          C_NAVY if i < 3 else C_MUTED, True),
            (grade_txt,                grade_color(rd["grade"]),      True),
            (rd["severity"],           severity_color(rd["severity"]),False),
            (rd["size"],               C_BLACK,                       False),
            (rd["location"],           C_BLACK,                       False),
            (f"{r['similarity']}%",    C_NAVY if r["similarity"]>=80 else C_BLACK, True),
            (f"{r['embedding_sim']}%", C_MUTED,                       False),
            (f"{r['attr_sim']}%",      C_MUTED,                       False),
        ]
        for w, (v, vc, bold) in zip(cols, vals):
            cv.setFont("Helvetica-Bold" if bold else "Helvetica", 8)
            cv.setFillColor(vc)
            cv.drawCentredString(rx + w/2, y - 3.5*mm, v); rx += w
        y -= 6.5*mm

    y -= 3*mm
    cv.setFillColor(colors.HexColor("#eaf7f0"))
    cv.rect(ML, y - 5*mm, 70*mm, 5.5*mm, fill=1, stroke=0)
    cv.setFont("Helvetica", 7); cv.setFillColor(C_GREEN)
    cv.drawString(ML + 2*mm, y - 2.5*mm, "Green rows / ✓ = grade matches query scan")
    y -= 12*mm

    y = section_heading(cv, y, "Attribute Score Breakdown  —  Top 5 Retrieved Cases")
    cv.setFont("Helvetica", 7.5); cv.setFillColor(C_MUTED)
    cv.drawString(ML, y + 4*mm,
        "Weighted attribute similarity scores  (Grade 35%  |  Severity 30%  |  Size 20%  |  Location 15%)")
    y -= 11*mm

    b_cols = [12*mm, 34*mm, 34*mm, 30*mm, 30*mm, 30*mm]
    b_hdrs = ["Rank","Grade Score","Severity Score","Size Score","Location Score","Overall"]
    cv.setFillColor(C_NAVY_LT)
    cv.rect(ML, y - 5.5*mm, CW, 6.5*mm, fill=1, stroke=0)
    cv.setFillColor(C_WHITE); cv.setFont("Helvetica-Bold", 7.5)
    bx = ML
    for w, h_lbl in zip(b_cols, b_hdrs):
        cv.drawCentredString(bx + w/2, y - 3*mm, h_lbl); bx += w
    y -= 5.5*mm

    for i, r in enumerate(rows[:5]):
        b = r["breakdown"]
        cv.setFillColor(colors.HexColor("#f8fbff") if i % 2 == 0 else C_WHITE)
        cv.rect(ML, y - 6*mm, CW, 6.5*mm, fill=1, stroke=0)
        cv.setStrokeColor(C_BORDER); cv.setLineWidth(0.3)
        cv.line(ML, y - 6*mm, ML + CW, y - 6*mm)
        bx = ML
        bvals = [
            (f"#{r['rank']}",       C_NAVY,  True),
            (f"{b['grade']}%",      C_GREEN if b["grade"]    >= 80 else C_ORANGE, False),
            (f"{b['severity']}%",   C_GREEN if b["severity"] >= 80 else C_ORANGE, False),
            (f"{b['size']}%",       C_GREEN if b["size"]     >= 80 else C_ORANGE, False),
            (f"{b['location']}%",   C_GREEN if b["location"] >= 80 else C_ORANGE, False),
            (f"{r['similarity']}%", C_NAVY,  True),
        ]
        for w, (v, vc, bold) in zip(b_cols, bvals):
            cv.setFont("Helvetica-Bold" if bold else "Helvetica", 8)
            cv.setFillColor(vc)
            cv.drawCentredString(bx + w/2, y - 3.5*mm, v); bx += w
        y -= 6.5*mm

    y -= 10*mm
    y = section_heading(cv, y, "Limitations & Disclaimer")
    lbl     = "Important:"
    lbl_w   = cv.stringWidth(lbl, "Helvetica-Bold", 8)
    txt     = ("AI predictions are probabilistic and subject to variability based on image quality, "
               "acquisition parameters, and case complexity. This report must be reviewed and verified "
               "by a qualified radiologist or neurologist before any clinical decision is made. "
               "NeuroInsight is a research tool and is not approved for diagnostic use.")
    cv.setFont("Helvetica-Bold", 8); cv.setFillColor(C_NAVY)
    cv.drawString(ML, y, lbl)
    wrap_text(cv, txt, ML + lbl_w + 3*mm, y, CW - lbl_w - 3*mm,
              font="Helvetica", size=8, color=C_BLACK, line_h=4.5*mm)


# ── Public API ──────────────────────────────────────────────────────────────

def generate_pdf_report(result: dict) -> bytes:
    buf       = io.BytesIO()
    now       = datetime.now()
    now_str   = now.strftime("%d %b %Y  %H:%M")
    report_id = f"NI-{now.strftime('%Y%m%d')}-{now.strftime('%H%M%S')}"

    # Patient data from frontend modal (falls back to defaults if missing)
    patient = result.get("patient", {})

    tumor_present = result["diagnosis"].get("tumor_present", True)

    cv = canvas.Canvas(buf, pagesize=A4)
    cv.setTitle("NeuroInsight — Radiology Report")
    cv.setAuthor("NeuroInsight AI v1.0")
    cv.setSubject("AI-Assisted Brain Tumour Analysis")
    cv.setCreator("NeuroInsight v1.0 — ResNet-50 Phase II")

    if tumor_present:
        # ── 2-page report: findings + retrieval ──
        total_pages = 2
        draw_page_frame(cv, 1, total_pages, report_id)
        draw_page1_tumor(cv, result, patient, report_id, now_str)
        cv.showPage()
        draw_page_frame(cv, 2, total_pages, report_id)
        draw_page2(cv, result, report_id)
        cv.showPage()
    else:
        # ── 1-page report: no-tumor summary ──
        total_pages = 1
        draw_page_frame(cv, 1, total_pages, report_id)
        draw_page1_no_tumor(cv, result, patient, report_id, now_str)
        cv.showPage()

    cv.save()
    buf.seek(0)
    return buf.read()


# ── Test harness ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Test 1: Tumor detected
    dummy_tumor = {
        "inference_ms": 143,
        "patient": {
            "name":       "Priya Sharma",
            "dob":        "12 / 03 / 1985",
            "gender":     "Female",
            "patientId":  "NI-2026-0042",
            "physician":  "Dr. Karra Renu",
            "history":    "Presenting with persistent headaches and visual disturbances for 3 weeks.",
            "sequences":  "T1 / T2 / FLAIR Axial",
        },
        "diagnosis": {
            "grade": "HGG", "severity": "high",
            "size": "large", "location": "right",
            "tumor_present": True, "tumor_confidence": 94.2,
        },
        "confidence": { "grade": 87.3, "severity": 91.2, "size": 78.5, "location": 83.1 },
        "retrieval": {
            "total_searched": 48, "returned": 10,
            "results": [
                {
                    "rank": i + 1,
                    "similarity":    round(92 - i * 3.1, 1),
                    "embedding_sim": round(88 - i * 2.8, 1),
                    "attr_sim":      round(85 - i * 3.5, 1),
                    "breakdown": {
                        "grade": round(100-i*2,1), "severity": round(90-i*5,1),
                        "size":  round(80-i*4,1),  "location": round(75-i*3,1),
                    },
                    "diagnosis": {
                        "grade":    ["HGG","HGG","HGG","HGG","HGG","LGG","HGG","HGG","LGG","HGG"][i],
                        "severity": ["high","high","medium","high","low","medium","high","medium","high","low"][i],
                        "size":     ["large","large","medium","large","small","medium","large","medium","large","small"][i],
                        "location": ["right","left","right","bilateral","right","left","right","right","left","bilateral"][i],
                    },
                } for i in range(10)
            ],
        },
        "images": {"original": "", "heatmap": ""},
    }

    # Test 2: No tumor
    dummy_no_tumor = {
        "inference_ms": 98,
        "patient": {
            "name":      "Rahul Verma",
            "dob":       "05 / 07 / 1992",
            "gender":    "Male",
            "patientId": "NI-2026-0043",
            "physician": "Dr. Alamuri",
            "history":   "Routine MRI follow-up. No current neurological complaints.",
            "sequences": "T1 / T2 Axial",
        },
        "diagnosis": {
            "tumor_present": False,
            "tumor_confidence": 96.8,
            "grade": "N/A", "severity": "low",
            "size": "N/A", "location": "N/A",
        },
        "confidence": { "grade": 0, "severity": 0, "size": 0, "location": 0 },
        "retrieval":  { "total_searched": 0, "returned": 0, "results": [] },
        "images":     { "original": "", "heatmap": "" },
    }

    with open("report_tumor.pdf", "wb") as f:
        f.write(generate_pdf_report(dummy_tumor))
    print("Saved → report_tumor.pdf")

    with open("report_no_tumor.pdf", "wb") as f:
        f.write(generate_pdf_report(dummy_no_tumor))
    print("Saved → report_no_tumor.pdf")