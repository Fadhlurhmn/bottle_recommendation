# -*- coding: utf-8 -*-
"""
app_yoloworld.py
================
API Rekomendasi Botol, Cup, Tray, dan Closure menggunakan:
1. YOLO-World v2 (yolov8m-worldv2.pt) untuk open-vocabulary detection & klasifikasi kategori.
2. GrabCut + morphological cleaning untuk segmentasi & cropping objek ke latar putih.
3. DINOv2 + 9 Shape Features (777-dim) untuk CBIR Image Search.
4. SigLIP + BM25 + Multilingual-E5 (RRF Fusion) untuk TBIR Text Search.
5. Qdrant Merged Collection ("bottle-recommendation-qdrant-merged") dengan auto-filtering berdasarkan 'Type' hasil deteksi YOLO-World.
"""

import os
import re
import cv2
import json
import base64
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import uvicorn
import httpx

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from ultralytics import YOLOWorld
from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# =============================================================================
# KONFIGURASI
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DINOV2_NAME = "facebook/dinov2-base"
TBIR_MODEL_NAME = "google/siglip-base-patch16-224"
E5_MODEL_NAME = "intfloat/multilingual-e5-base"

# YOLO-World configuration
YOLO_WORLD_MODEL_NAME = "yolov8m-worldv2.pt"
YOLO_CONF = 0.05
YOLO_IMGSZ = 640

# Target Qdrant merged database (Local folder path atau Qdrant Cloud URL)
QDRANT_PATH = os.environ.get("QDRANT_URL", "bottle-recommendation-qdrant-merged")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", None)
COLLECTION_NAME = "bottle_collection_storage"
FINAL_K = 15

INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# =============================================================================
# DICTIONARY KELAS YOLO-WORLD & MAPPING TYPE
# =============================================================================
TEXTS = {
    'bottle': [
        # --- Beverage & General Liquid ---
        "plastic beverage bottle",          # Botol minuman umum (kopi, teh, jus, susu)
        "water bottle",                     # Botol air mineral / AMDK
        "clear plastic bottle",             # Botol bening transparan
        "flat plastic flask bottle",        # Botol pipih/gepeng (cold brew, bir pletok)

        # --- Food ---
        "plastic food bottle",              # Botol makanan (kecap, saus)
        "plastic squeeze bottle",           # Botol pencet (madu, saus, mayonaise)
        "plastic cooking oil bottle",       # Botol minyak goreng / minyak sayur
        "plastic spice jar",                # Jar bumbu dapur, cabe bubuk, abon

        # --- Chemical ---
        "plastic chemical bottle",          # Botol pestisida, pupuk cair, bahan kimia
        "battery water bottle",             # Botol air aki / air accu

        # --- Homecare ---
        "trigger spray bottle",             # Botol pembersih kaca & disinfektan spray
        "dish soap bottle",                 # Botol sabun cuci piring (dishwash)
        "cleaning product bottle",          # Botol pembersih rumah tangga umum

        # --- Personal Care ---
        "plastic cosmetic bottle",          # Botol kosmetik umum (face treatment, toner)
        "shampoo bottle",                   # Botol shampoo & sabun mandi cair
        "lotion pump bottle",               # Botol lotion / sabun cair pump
        "baby oil bottle",                  # Botol minyak telon, minyak kayu putih, baby oil
        "powder shaker bottle",             # Botol bedak tabur

        # --- Pharmaceutical ---
        "liquid medicine bottle",           # Botol sirup obat batuk / OBH
        "plastic pill bottle",              # Botol kapsul, tablet, herbal, vitamin
        "dropper bottle",                   # Botol tetes (food coloring, essence, essential oil)

        # --- Jar & Pot ---
        "plastic cookie jar",               # Toples kue kering, nastar, wafer stik
        "plastic food jar",                 # Toples selai, sambal, bumbu dasar
        "plastic chili sauce jar",          # Jar sambal & bawang goreng
        "cosmetic cream jar",               # Pot krim ceper (body butter, lulur, skincare)
        "candy jar",                        # Toples permen, gulali, snack
        "clear plastic jar",                # Toples silinder bening transparan
        "plastic jar",                      # Kelas fallback toples umum
    ],

    'closure': [
        # --- Standard & Screw Caps (Umum & Beverage) ---
        "plastic bottle cap",               # Tutup botol ulir plastik standar (beverage, AMDK)
        "tamper evident bottle cap",        # Tutup botol segel putus (minuman segel, sirup)
        "sports bottle cap",                # Tutup sport / push-pull drink cap

        # --- Dispensing & Food Caps ---
        "flip top cap",                     # Tutup flip-top (shampoo, saus, lotion)
        "disc top cap",                     # Tutup tekan satu sisi / disc-top (toner, sabun)
        "push pull cap",                    # Tutup tarik cuci piring / dishwash
        "sauce nozzle cap",                 # Tutup moncong lancip (botol saus/kecap/madu)
        "seasoning shaker lid",             # Tutup bumbu tabur / lubang garam-merica

        # --- Sprayers & Pumps (Homecare & Personal Care) ---
        "trigger sprayer",                  # Kepala spray pembersih kaca & disinfektan (model pistol)
        "fine mist spray pump",             # Kepala spray halus / mist (face mist, parfum, body mist)
        "lotion dispenser pump",            # Kepala pompa lotion, sabun mandi, shampoo, sirup pump
        "foam pump",                        # Kepala pompa busa / foaming soap pump

        # --- Pharmaceutical & Chemical Closures ---
        "child resistant cap",              # Tutup pengaman anak (tekan lalu putar untuk obat/kimia)
        "dropper pipet cap",                # Tutup pipet tetes (essential oil, serum obat/pewarna)
        "medicine measuring cup",           # Tutup takar bening untuk sirup obat batuk / OBH

        # --- Jar & Wide-Mouth Lids ---
        "wide mouth jar lid",               # Tutup toples kue kering & sambal mulut lebar
        "cosmetic cream jar lid",           # Tutup pot krim ceper
        "aluminum jar lid",                 # Tutup toples aluminium / metal
    ],

    'tray': [
        "food packaging plastic tray",      # Tray mika / baki makanan plastik
        "cake plastic packaging",           # Mika / kemasan kue & pastry
    ],

    'cup': [
        "plastic beverage cup",             # Gelas minuman plastik umum (kopi, boba, teh)
        "disposable plastic cup",           # Cup plastik sekali pakai
        "pudding cup",                      # Cup mini puding / fla
        "dessert cup",                      # Cup yogurt, dessert, es krim
        "oval plastic cup",                 # Cup bentuk u-shape / oval kekinian
        "clear plastic cup",                # Gelas plastik transparan / bening
    ],

    'lid': [
        "flat cup lid",                     # Tutup cup datar (flat lid)
        "dome cup lid",                     # Tutup cup cembung (dome lid untuk es krim/boba)
        "strawless sip lid",                # Tutup cup lubang teguk tanpa sedotan (kopi kekinian)
        "cup sealer film",                  # Plastik seal press penutup cup
        "small dessert cup lid",            # Tutup ceper cup puding / yogurt mini
        "plastic tray lid",                 # Tutup tray / baki makanan plastik bening
        "plastic cup lid",                  # Kelas fallback umum untuk tutup cup
    ],
}

# Flatten kelas dan petakan ke Type
ALL_CLASSES = []
CLASS_TO_TYPE = {}
for category_type, prompt_list in TEXTS.items():
    for prompt_text in prompt_list:
        ALL_CLASSES.append(prompt_text)
        CLASS_TO_TYPE[prompt_text] = category_type

# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================
app = FastAPI(
    title="Product Recommender API (YOLO-World + Qdrant Merged)",
    version="2.0.0",
    description="API pencarian visual dan teks untuk Botol, Cup, Tray, dan Closure dengan segmentasi YOLO-World."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model & DB handlers
yolo_world_model = None
dino_model = None
tbir_tokenizer = None
tbir_model = None
qdrant_client = None
lr_models: Dict[str, Any] = {}
xgb_models = lr_models  # alias backward compatibility
e5_model = None

db_metadata_docs = []
db_qdrant_ids = []
db_payloads = []
bm25_index = None
e5_vectors = None


def build_structured_product_text(p: dict) -> str:
    """
    Membangun representasi teks terstruktur (Key-Value) yang optimal
    untuk Multilingual-E5 (semantik) dan BM25 (leksikal).
    """
    if not p:
        return ""
    parts = []

    def add(label: str, key: str, unit: str = ""):
        val = p.get(key)
        if val is not None:
            s_val = str(val).strip()
            if s_val and s_val.lower() != "none":
                if unit:
                    parts.append(f"{label}: {s_val} {unit}")
                else:
                    parts.append(f"{label}: {s_val}")

    add("produk", "ProductName")
    add("tipe", "Type")
    add("kategori", "CategoryName")
    add("warna", "ColorName")
    add("material", "Material")
    add("harga", "Price")
    add("volume", "ProductVolume", "ml")
    add("neck", "ProductNeck")
    add("dimensi", "PackagingDimension")
    add("berat", "Weight", "gram")
    add("tinggi", "Height", "mm")
    add("panjang", "Length", "mm")
    add("lebar", "Width", "mm")
    add("fungsi", "ProductFunction")
    add("keamanan", "ProductSafety")
    add("tags", "Tags")
    add("catatan", "Notes")

    return " | ".join(parts).lower()


@app.on_event("startup")
async def startup_event():
    global yolo_world_model, dino_model, tbir_tokenizer, tbir_model, qdrant_client
    global lr_models, xgb_models, e5_model, db_metadata_docs, db_qdrant_ids, db_payloads
    global bm25_index, e5_vectors

    print(f"[STARTUP] Device: {DEVICE}")

    # 1. Load YOLO-World
    print(f"[STARTUP] Loading YOLO-World ({YOLO_WORLD_MODEL_NAME})...")
    yolo_world_model = YOLOWorld(YOLO_WORLD_MODEL_NAME)
    yolo_world_model.set_classes(ALL_CLASSES)
    # Warm-up predict
    yolo_world_model.predict(
        np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), np.uint8),
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        device=DEVICE,
        verbose=False
    )
    print(f"[STARTUP] YOLO-World siap dengan {len(ALL_CLASSES)} kelas di 5 kategori.")

    # 2. Load DINOv2
    print("[STARTUP] Loading CBIR Model (DINOv2)...")
    dino_model = AutoModel.from_pretrained(DINOV2_NAME).to(DEVICE)
    dino_model.eval()

    # 3. Load SigLIP
    print("[STARTUP] Loading TBIR Model (SigLIP)...")
    tbir_tokenizer = AutoTokenizer.from_pretrained(TBIR_MODEL_NAME)
    tbir_model = AutoModel.from_pretrained(TBIR_MODEL_NAME).to(DEVICE)
    tbir_model.eval()

    # 4. Connect to Qdrant Merged (Local atau Cloud)
    if QDRANT_PATH.startswith("http://") or QDRANT_PATH.startswith("https://"):
        print(f"[STARTUP] Connecting to Qdrant Cloud di '{QDRANT_PATH}'...")
        qdrant_client = QdrantClient(url=QDRANT_PATH, api_key=QDRANT_API_KEY, timeout=60.0)
    else:
        print(f"[STARTUP] Connecting to Qdrant Local di '{QDRANT_PATH}'...")
        qdrant_client = QdrantClient(path=QDRANT_PATH)

    # 5. Load / Auto-train Logistic Regression Rerankers per Type
    print("[STARTUP] Memeriksa model Logistic Regression Reranker per Type...")
    TYPES_TO_TRAIN = ["bottle", "cup", "tray"]
    missing_types = [t for t in TYPES_TO_TRAIN if not os.path.exists(f"lr_model_{t}.pkl")]

    if missing_types:
        print(f"[STARTUP] Model Logistic Regression untuk Type {missing_types} belum ditemukan di disk.")
        print("[STARTUP] Mengambil vektor CBIR dari Qdrant Merged untuk melatih model per-type...")
        from collections import defaultdict
        type_train_data = defaultdict(lambda: {"X": [], "y": []})

        offset = None
        while True:
            records, next_page = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=1000,
                offset=offset,
                with_vectors=["cbir"],
                with_payload=["Type", "class_name"]
            )
            for r in records:
                p = r.payload
                vec = r.vector.get("cbir") if r.vector else None
                if p and vec is not None:
                    t = str(p.get("Type", "")).lower().strip()
                    cls_name = p.get("class_name", "")
                    if t in TYPES_TO_TRAIN and cls_name:
                        type_train_data[t]["X"].append(vec)
                        type_train_data[t]["y"].append(cls_name)

            if next_page is None:
                break
            offset = next_page

        for t in TYPES_TO_TRAIN:
            model_file = f"lr_model_{t}.pkl"
            if os.path.exists(model_file):
                try:
                    with open(model_file, "rb") as f:
                        lr_models[t] = pickle.load(f)
                    print(f"[STARTUP] Model existing '{model_file}' berhasil dimuat.")
                    continue
                except Exception as e:
                    print(f"[STARTUP] Gagal memuat '{model_file}' ({e}), melakukan train ulang...")

            d = type_train_data[t]
            if not d["X"]:
                print(f"[STARTUP] Data untuk Type '{t}' kosong, melewati pelatihan.")
                continue

            X = np.array(d["X"], dtype=np.float32)
            le = LabelEncoder()
            y = le.fit_transform(d["y"])
            n_classes = len(le.classes_)

            if n_classes > 1:
                print(f"[STARTUP] Melatih Logistic Regression untuk Type '{t}' ({len(X)} sampel, {n_classes} kelas)...")
                clf = LogisticRegression(
                    class_weight="balanced",
                    C=1.0,
                    max_iter=1000,
                    solver="lbfgs",
                    n_jobs=4
                )
                clf.fit(X, y)
                with open(model_file, "wb") as f:
                    pickle.dump((clf, le), f)
                lr_models[t] = (clf, le)
                print(f"[STARTUP] ✅ Model '{model_file}' berhasil dilatih dan disimpan.")
            else:
                print(f"[STARTUP] Type '{t}' hanya memiliki {n_classes} kelas, melewati pelatihan (fallback cosine).")
    else:
        for t in TYPES_TO_TRAIN:
            model_file = f"lr_model_{t}.pkl"
            try:
                with open(model_file, "rb") as f:
                    lr_models[t] = pickle.load(f)
                print(f"[STARTUP] Model '{model_file}' berhasil dimuat.")
            except Exception as e:
                print(f"[STARTUP] Peringatan: Gagal memuat '{model_file}': {e}")

    # 6. Load E5 & Metadata Indexing (dengan Cache Disk)
    print("[STARTUP] Loading E5 Model for Hybrid Text Search...")
    e5_model = SentenceTransformer(E5_MODEL_NAME, device=DEVICE)

    CACHE_EMB_FILE = "e5_vectors_merged.npy"
    CACHE_META_FILE = "metadata_cache_merged.pkl"

    if os.path.exists(CACHE_EMB_FILE) and os.path.exists(CACHE_META_FILE):
        print("[STARTUP] Memuat BM25 & E5 index dari disk cache...")
        with open(CACHE_META_FILE, "rb") as f:
            db_metadata_docs, db_qdrant_ids, db_payloads = pickle.load(f)
        e5_vectors = np.load(CACHE_EMB_FILE)
        print(f"[STARTUP] Cache berhasil dimuat ({len(db_metadata_docs)} item).")
    else:
        print("[STARTUP] Membaca metadata dari Qdrant Merged untuk BM25 & E5...")
        offset = None
        while True:
            records, next_page = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_vectors=False,
                with_payload=True
            )
            for r in records:
                p = r.payload
                if p:
                    text_doc = build_structured_product_text(p)
                    db_metadata_docs.append(text_doc)
                    db_qdrant_ids.append(r.id)
                    db_payloads.append(p)

            if next_page is None:
                break
            offset = next_page

        if db_metadata_docs:
            print(f"[STARTUP] Meng-encode {len(db_metadata_docs)} dokumen dengan E5 (batch_size=64)...")
            e5_passages = [f"passage: {doc}" for doc in db_metadata_docs]
            e5_vectors = e5_model.encode(
                e5_passages,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            np.save(CACHE_EMB_FILE, e5_vectors)
            with open(CACHE_META_FILE, "wb") as f:
                pickle.dump((db_metadata_docs, db_qdrant_ids, db_payloads), f)
            print("[STARTUP] Cache E5 & metadata berhasil disimpan ke disk.")

    tokenized_docs = [doc.split() for doc in db_metadata_docs]
    bm25_index = BM25Okapi(tokenized_docs) if tokenized_docs else None

    print(f"[STARTUP] Index BM25 & E5 siap dengan {len(db_metadata_docs)} item.")
    print("[STARTUP] Sistem YOLO-World + Qdrant Merged siap digunakan!\n")


# =============================================================================
# DETEKSI YOLO-WORLD & SEGMENTASI GRABCUT
# =============================================================================

def detect(img_rgb: np.ndarray) -> List[Dict[str, Any]]:
    """
    Deteksi objek menggunakan YOLO-World open-vocabulary.
    img_rgb: citra RGB (H, W, 3).
    Returns list of dict sorted by score descending.
    """
    # Ultralytics memproses format BGR
    r = yolo_world_model.predict(
        img_rgb[..., ::-1],
        conf=YOLO_CONF,
        imgsz=YOLO_IMGSZ,
        device=DEVICE,
        verbose=False
    )[0]

    dets = []
    if r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for b, s, c in zip(boxes, confs, classes):
            cls_idx = int(c)
            if 0 <= cls_idx < len(ALL_CLASSES):
                text_label = ALL_CLASSES[cls_idx]
                type_category = CLASS_TO_TYPE.get(text_label, "unknown")
                dets.append({
                    "box": b.astype(float),
                    "score": float(s),
                    "text": text_label,
                    "type": type_category
                })

    return sorted(dets, key=lambda d: -d["score"])


def grabcut_from_box(img_rgb: np.ndarray, box: np.ndarray, pad: float = 0.05, max_side: int = 512, iters: int = 4):
    """
    Eksekusi GrabCut di dalam bounding box yang diberikan dengan padding.
    """
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(W, x2 + px), min(H, y2 + py)

    crop = img_rgb[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return None, 0.0

    sc = min(1.0, max_side / max(crop.shape[:2]))
    small = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)

    mask = np.zeros(small.shape[:2], np.uint8)
    m = max(2, int(min(small.shape[:2]) * 0.03))
    rect = (m, m, small.shape[1] - 2 * m, small.shape[0] - 2 * m)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), mask, rect, bgd, fgd,
                    iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None, 0.0

    fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    fg = cv2.resize(fg, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    ratio = float((fg > 0).mean())
    if ratio < 0.25 or ratio > 0.95:
        return None, ratio

    full = np.zeros((H, W), np.uint8)
    full[y1:y2, x1:x2] = fg
    return full, ratio


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Bersihkan noise mask: ambil komponen terbesar, morph close, dan convex hull."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        mask = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    if cnts:
        cv2.drawContours(out, [cv2.convexHull(max(cnts, key=cv2.contourArea))], -1, 255, -1)
    return out


def mask_box_coverage(mask: np.ndarray, box: np.ndarray) -> float:
    """Hitung rasio cakupan mask terhadap bounding box."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return 0.0
    x1, y1, x2, y2 = box
    bbox_area = (xs.max() - xs.min()) * (ys.max() - ys.min())
    return bbox_area / ((x2 - x1) * (y2 - y1) + 1e-6)


def compose_crop_and_mask(img_rgb: np.ndarray, mask: np.ndarray, pad: float = 0.06, bg: int = 255):
    """
    Komposisikan crop objek di atas latar belakang putih dengan mask lembut (GaussianBlur).
    Returns (cropped_rgb, cropped_mask).
    """
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return img_rgb, mask

    y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
    py, px = int((y2 - y1) * pad), int((x2 - x1) * pad)
    y1, x1 = max(0, y1 - py), max(0, x1 - px)
    y2, x2 = min(img_rgb.shape[0], y2 + py), min(img_rgb.shape[1], x2 + px)

    soft = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32)[..., None] / 255.0
    comp = (img_rgb * soft + bg * (1.0 - soft)).astype(np.uint8)

    crop_rgb = comp[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    return crop_rgb, crop_mask


def fallback_segmentation(img_rgb: np.ndarray):
    """Fallback segmentasi jika tidak ada deteksi YOLO-World."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, mask_inv = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    mask_bin = cv2.bitwise_not(mask_inv)
    return img_rgb, mask_bin


# =============================================================================
# FEATURE EXTRACTION (CBIR & TBIR)
# =============================================================================

def extract_shape_features(mask_bin: np.ndarray) -> np.ndarray:
    """Ekstraksi 9 fitur bentuk geometri dari mask biner."""
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(9, dtype=np.float32)

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)

    aspect_ratio = float(h) / (w + 1e-6)
    extent = float(area) / (w * h + 1e-6)
    hull = cv2.convexHull(c)
    solidity = float(area) / (cv2.contourArea(hull) + 1e-6)
    compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-6)

    y25 = min(int(y + 0.25 * h), mask_bin.shape[0] - 1)
    y50 = min(int(y + 0.50 * h), mask_bin.shape[0] - 1)
    y75 = min(int(y + 0.75 * h), mask_bin.shape[0] - 1)
    w25, w50, w75 = np.sum(mask_bin[y25, :] > 0), np.sum(mask_bin[y50, :] > 0), np.sum(mask_bin[y75, :] > 0)
    neck_ratio = float(w25) / (w75 + 1e-6)
    waist_ratio = float(w50) / (w75 + 1e-6)

    moments = cv2.moments(c)
    hu = cv2.HuMoments(moments).flatten()
    log_hu = -np.sign(hu[:2]) * np.log10(np.abs(hu[:2]) + 1e-12)

    shape_vec = np.array([aspect_ratio, extent, solidity, compactness, neck_ratio, waist_ratio, log_hu[0], log_hu[1], 0.0], dtype=np.float32)
    norm = np.linalg.norm(shape_vec)
    return shape_vec / (norm + 1e-8)


def pad_square(arr: np.ndarray) -> np.ndarray:
    """Pad array ke bentuk bujur sangkar (square) dengan latar hitam."""
    h, w = arr.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=arr.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0+h, x0:x0+w] = arr
    return canvas


@torch.no_grad()
def extract_cbir_vector(crop_rgb: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    """Ekstraksi vector CBIR gabungan (DINOv2 768-dim + 9-dim shape = 777-dim)."""
    sq_rgb = cv2.resize(pad_square(crop_rgb), (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    norm_rgb = (sq_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(norm_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    out = dino_model(pixel_values=tensor).last_hidden_state
    dino_feat = F.normalize(out[:, 0, :], p=2, dim=1)[0].cpu().numpy().astype(np.float32)

    shape_feat = extract_shape_features(crop_mask)
    w_shape = 0.25
    cbir_fused = np.hstack([np.sqrt(1.0 - w_shape) * dino_feat, np.sqrt(w_shape) * shape_feat])
    return (cbir_fused / (np.linalg.norm(cbir_fused) + 1e-8)).astype(np.float32)


@torch.no_grad()
def extract_tbir_vector(text_query: str) -> np.ndarray:
    """Ekstraksi vector TBIR (SigLIP 768-dim)."""
    inputs = tbir_tokenizer([text_query], padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
    if hasattr(tbir_model, 'get_text_features'):
        out = tbir_model.get_text_features(**inputs)
    else:
        out = tbir_model(**inputs)

    if hasattr(out, 'text_embeds') and out.text_embeds is not None:
        out = out.text_embeds
    elif hasattr(out, 'pooler_output') and out.pooler_output is not None:
        out = out.pooler_output
    elif hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
        out = out.last_hidden_state.mean(dim=1)

    out = F.normalize(out, p=2, dim=1)[0].cpu().numpy().astype(np.float32)
    return out


# =============================================================================
# API SCHEMAS & HELPERS
# =============================================================================

class MockPoint:
    def __init__(self, _id, score, payload):
        self.id = _id
        self.score = score
        self.payload = payload


class SearchResult(BaseModel):
    id: Any
    score: float
    product_image_url: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    message: str
    detected_type: Optional[str] = None
    detected_label: Optional[str] = None
    detection_score: Optional[float] = None
    segmentation_base64: Optional[str] = None
    results: List[SearchResult]


class ImageSearchPayload(BaseModel):
    image: str
    type: Optional[str] = None           # Override manual jika diinginkan ('bottle', 'tray', 'cup', dll.)
    filter_by_type: bool = True          # Otomatis filter Qdrant berdasarkan Type hasil YOLO-World


def image_to_base64(img_arr: np.ndarray) -> Optional[str]:
    """Konversi numpy array RGB/BGR ke Base64 JPEG string."""
    if img_arr is None or img_arr.size == 0:
        return None
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')


def format_api_results(res_list) -> List[SearchResult]:
    api_results = []
    for r in res_list:
        payload = r.payload or {}
        img_url = payload.get("ProductImageUrl", "")
        if not img_url:
            img_url = payload.get("image_path", "")

        api_results.append(SearchResult(
            id=r.id,
            score=float(r.score),
            product_image_url=img_url,
            metadata=payload
        ))
    return api_results


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
def api_root():
    return {
        "status": "online",
        "service": "Product Recommendation API (YOLO-World + Qdrant Merged)",
        "models": {
            "yolo_world": YOLO_WORLD_MODEL_NAME,
            "cbir": DINOV2_NAME,
            "tbir": TBIR_MODEL_NAME
        },
        "target_database": QDRANT_PATH,
        "supported_types": list(TEXTS.keys())
    }


@app.get("/health")
def api_health():
    models_ready = bool(
        yolo_world_model is not None and
        dino_model is not None and
        tbir_model is not None and
        qdrant_client is not None
    )
    return {
        "status": "healthy" if models_ready else "starting",
        "models_loaded": models_ready,
        "database": "connected" if qdrant_client is not None else "disconnected"
    }


def execute_cbir_search(contents: bytes, manual_type: Optional[str] = None, filter_by_type: bool = True) -> SearchResponse:
    """Eksekusi pipeline CBIR YOLO-World + GrabCut + DINOv2 + Qdrant + XGBoost dari byte gambar."""
    np_arr = np.frombuffer(contents, np.uint8)
    input_img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if input_img_bgr is None:
        raise HTTPException(status_code=400, detail="Gambar tidak valid atau corrupt.")

    image_rgb = cv2.cvtColor(input_img_bgr, cv2.COLOR_BGR2RGB)
    H, W = image_rgb.shape[:2]

    # 1. Deteksi YOLO-World
    dets = detect(image_rgb)

    detected_type = None
    detected_label = None
    detection_score = 0.0
    used_fallback = False

    if dets:
        best = dets[0]
        detected_label = best["text"]
        detected_type = best["type"]
        detection_score = best["score"]
        box = best["box"]

        # 2. GrabCut dari bounding box
        gc_mask, ratio = grabcut_from_box(image_rgb, box)
        if gc_mask is not None:
            gc_mask = clean_mask(gc_mask)
            if mask_box_coverage(gc_mask, box) < 0.75:
                gc_mask = None

        # Fallback jika GrabCut gagal
        if gc_mask is None:
            used_fallback = True
            gc_mask = np.zeros((H, W), np.uint8)
            x1, y1, x2, y2 = box.astype(int)
            gc_mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = 255

        crop_rgb, crop_mask = compose_crop_and_mask(image_rgb, gc_mask)
    else:
        # Fallback keseluruhan jika YOLO-World tidak menemukan objek
        used_fallback = True
        crop_rgb, crop_mask = fallback_segmentation(image_rgb)

    # 3. Ekstrak vektor CBIR (777-dim)
    cbir_vec = extract_cbir_vector(crop_rgb, crop_mask)

    # 4. Bangun Qdrant Filter berdasarkan Type
    filter_type = manual_type.lower().strip() if manual_type else detected_type

    query_filter = None
    if filter_by_type and filter_type:
        query_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="Type",
                    match=qmodels.MatchValue(value=filter_type)
                )
            ]
        )

    # 5. Query ke Qdrant
    res_qdrant = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=cbir_vec.tolist(),
        using="cbir",
        limit=50,
        query_filter=query_filter,
        with_payload=True
    ).points

    # Fallback tanpa filter jika hasil kosong
    if not res_qdrant and query_filter is not None:
        res_qdrant = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=cbir_vec.tolist(),
            using="cbir",
            limit=50,
            query_filter=None,
            with_payload=True
        ).points

    # 6. Reranking (Logistic Regression per Type + Cosine Score)
    res = []
    if res_qdrant:
        target_type = filter_type
        if not target_type and res_qdrant:
            target_type = str(res_qdrant[0].payload.get("Type", "")).lower().strip()

        active_model = lr_models.get(target_type) if target_type else None

        if active_model is not None:
            clf_type, le_type = active_model
            try:
                proba = clf_type.predict_proba(cbir_vec.reshape(1, -1))[0]
                class_to_prob = {str(c): float(p) for c, p in zip(le_type.classes_, proba)}

                cos_scores = np.array([r.score for r in res_qdrant], dtype=np.float32)
                lo, hi = cos_scores.min(), cos_scores.max()
                nc = np.zeros_like(cos_scores) if (hi - lo < 1e-8) else (cos_scores - lo) / (hi - lo)

                p_items = np.array([class_to_prob.get(str(r.payload.get('class_name', '')), 0.0) for r in res_qdrant], dtype=np.float32)

                alpha = 0.5
                final_scores = alpha * nc + (1.0 - alpha) * p_items + 1e-4 * nc

                order = np.argsort(-final_scores)
                res = [res_qdrant[i] for i in order[:FINAL_K]]
                for i in range(len(res)):
                    res[i].score = float(final_scores[order[i]])
            except Exception as e:
                print(f"[CBIR RERANK] Peringatan: Gagal reranking Logistic Regression ({e}), fallback ke cosine score.")
                res = res_qdrant[:FINAL_K]
        else:
            res = res_qdrant[:FINAL_K]

    if not res:
        return SearchResponse(
            message="⚠️ Tidak ada produk yang cocok di database (CBIR).",
            detected_type=detected_type,
            detected_label=detected_label,
            detection_score=detection_score,
            segmentation_base64=image_to_base64(crop_rgb),
            results=[]
        )

    api_results = format_api_results(res)
    filter_info = f" [Filter Type: '{filter_type}']" if filter_type else ""
    return SearchResponse(
        message=f"✅ Pencarian Gambar Selesai! (Menampilkan {len(api_results)} hasil){filter_info}",
        detected_type=detected_type,
        detected_label=detected_label,
        detection_score=detection_score,
        segmentation_base64=image_to_base64(crop_rgb),
        results=api_results
    )


@app.post("/search/image", response_model=SearchResponse)
async def api_search_image(payload: ImageSearchPayload):
    """
    Endpoint CBIR berbasis JSON Payload (URL, Base64, atau File Path).
    Cocok untuk integrasi n8n atau API eksternal.
    """
    try:
        raw_img = payload.image.strip()
        contents = None
        if raw_img.startswith("http://") or raw_img.startswith("https://"):
            async with httpx.AsyncClient(timeout=30.0) as http_client:
                resp = await http_client.get(raw_img)
                if resp.status_code != 200:
                    raise HTTPException(status_code=400, detail="Gagal mengunduh gambar dari URL CDN.")
                contents = resp.content
        elif raw_img.startswith("data:image"):
            header, encoded = raw_img.split(",", 1)
            contents = base64.b64decode(encoded)
        elif os.path.exists(raw_img):
            with open(raw_img, "rb") as f:
                contents = f.read()
        else:
            try:
                contents = base64.b64decode(raw_img)
            except Exception:
                raise HTTPException(status_code=400, detail="Format gambar tidak dikenali (harus URL http, base64, atau path file).")

        return execute_cbir_search(contents, payload.type, payload.filter_by_type)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Terjadi Kesalahan (CBIR YOLO-World): {str(e)}")


@app.post("/search/image/upload", response_model=SearchResponse)
async def api_search_image_upload(
    file: UploadFile = File(...),
    type: Optional[str] = Form(None),
    filter_by_type: bool = Form(True)
):
    """
    Endpoint CBIR berbasis Multipart Form-Data Upload (untuk Streamlit UI / browser upload).
    """
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="File gambar kosong.")
        return execute_cbir_search(contents, type, filter_by_type)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Terjadi Kesalahan (CBIR Upload): {str(e)}")


@app.post("/search/text", response_model=SearchResponse)
async def api_search_text(request: Request):
    """
    Endpoint untuk mencari produk menggunakan teks (TBIR).
    Mendukung JSON Body (application/json) maupun Form-data (multipart/x-www-form-urlencoded).
    """
    content_type = request.headers.get("content-type", "").lower()
    query = ""
    t_filter = None

    if "application/json" in content_type:
        try:
            body = await request.json()
            query = body.get("text_query", "")
            t_filter = body.get("type_filter")
        except Exception:
            pass
    else:
        try:
            form = await request.form()
            query = form.get("text_query", "")
            t_filter = form.get("type_filter")
        except Exception:
            pass

    if not query:
        query = request.query_params.get("text_query", "")
        if not t_filter:
            t_filter = request.query_params.get("type_filter")

    if not query or not str(query).strip():
        raise HTTPException(status_code=400, detail="Deskripsi teks (text_query) tidak boleh kosong.")

    query = str(query).strip()
    if t_filter:
        t_filter = str(t_filter).strip()

    try:
        tbir_vec = extract_tbir_vector(query)

        # Bangun query filter jika type_filter disediakan
        q_filter = None
        if t_filter:
            q_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="Type",
                        match=qmodels.MatchValue(value=t_filter.lower())
                    )
                ]
            )

        # 1. Query visual text ke Qdrant (SigLIP)
        tbir_res = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=tbir_vec.tolist(),
            using="tbir",
            limit=50,
            query_filter=q_filter,
            with_payload=True
        ).points

        # 2. BM25 Lexical Ranking
        tokenized_query = query.lower().split()
        bm25_scores = bm25_index.get_scores(tokenized_query) if bm25_index else np.zeros(len(db_metadata_docs))

        # 3. E5 Semantic Dense Ranking
        e5_vec = e5_model.encode([f"query: {query}"], convert_to_numpy=True, normalize_embeddings=True)[0]
        e5_scores = np.dot(e5_vectors, e5_vec) if e5_vectors is not None else np.zeros(len(db_metadata_docs))

        # 4. Filter indices jika type_filter aktif
        valid_indices = list(range(len(db_metadata_docs)))
        if t_filter:
            target_t = t_filter.lower()
            valid_indices = [
                i for i in valid_indices
                if db_payloads[i].get("Type", "").lower() == target_t
            ]

        if not valid_indices:
            valid_indices = list(range(len(db_metadata_docs)))

        # Ranks
        bm25_sub_scores = bm25_scores[valid_indices]
        bm25_order = np.argsort(-bm25_sub_scores)
        bm25_ranks = np.zeros(len(valid_indices), dtype=np.int32)
        bm25_ranks[bm25_order] = np.arange(1, len(valid_indices) + 1)

        e5_sub_scores = e5_scores[valid_indices]
        e5_order = np.argsort(-e5_sub_scores)
        e5_ranks = np.zeros(len(valid_indices), dtype=np.int32)
        e5_ranks[e5_order] = np.arange(1, len(valid_indices) + 1)

        # SigLIP ranks
        k_rrf = 60
        tbir_ranks = np.full(len(valid_indices), len(valid_indices) + 1, dtype=np.int32)
        id_to_sub_idx = {db_qdrant_ids[idx]: i for i, idx in enumerate(valid_indices)}

        for rank_i, r_p in enumerate(tbir_res):
            if r_p.id in id_to_sub_idx:
                tbir_ranks[id_to_sub_idx[r_p.id]] = rank_i + 1

        # RRF Calculation
        rrf_scores = (1.0 / (k_rrf + bm25_ranks)) + (1.0 / (k_rrf + e5_ranks)) + (1.0 / (k_rrf + tbir_ranks))
        final_order = np.argsort(-rrf_scores)[:FINAL_K]

        res = []
        for rank_pos in final_order:
            actual_idx = valid_indices[rank_pos]
            q_id = db_qdrant_ids[actual_idx]
            payload = db_payloads[actual_idx]
            score = float(rrf_scores[rank_pos])
            res.append(MockPoint(q_id, score, payload))

        if not res:
            return SearchResponse(
                message="⚠️ Tidak ada produk yang cocok di database (TBIR).",
                results=[]
            )

        api_results = format_api_results(res)
        return SearchResponse(
            message=f"✅ Pencarian Teks Selesai! (Menampilkan {len(api_results)} hasil)",
            results=api_results
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Terjadi Kesalahan (TBIR): {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app_yoloworld:app", host="0.0.0.0", port=8000, reload=True)
