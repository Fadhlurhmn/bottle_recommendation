import os
import re
import cv2
import json
import base64
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import uvicorn

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from fastembed import TextEmbedding, SparseTextEmbedding
from ultralytics import YOLO
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

# =============================================================================
# KONFIGURASI
# =============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DINOV2_NAME = "facebook/dinov2-base"
YOLO_MODEL_NAME = "yolo26m-seg.pt" 

TBIR_MODEL_NAME = "google/siglip-base-patch16-224"

QDRANT_PATH = "qdrant_bottle_recommendation"
COLLECTION_NAME = "bottle_collection"
FINAL_K = 10

INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================
app = FastAPI(title="Bottle Recommender API", version="1.0.0", description="API untuk pencarian botol (CBIR dan TBIR).")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variabel Global untuk Model & DB
yolo_model = None
dino_model = None
tbir_tokenizer = None
tbir_model = None
qdrant_client = None
lr_clf = None
label_encoder = None
e5_model = None
sparse_model = None

@app.on_event("startup")
async def startup_event():
    global yolo_model, dino_model, tbir_tokenizer, tbir_model, qdrant_client, lr_clf, label_encoder
    global e5_model, sparse_model

    print("[STARTUP] Loading YOLO Segmentation...")
    yolo_model = YOLO(YOLO_MODEL_NAME)

    print("[STARTUP] Loading CBIR Model (DINOv2)...")
    dino_model = AutoModel.from_pretrained(DINOV2_NAME).to(DEVICE)
    dino_model.eval()

    print("[STARTUP] Loading TBIR Model...")
    tbir_tokenizer = AutoTokenizer.from_pretrained(TBIR_MODEL_NAME)
    tbir_model = AutoModel.from_pretrained(TBIR_MODEL_NAME).to(DEVICE)
    tbir_model.eval()

    print(f"[STARTUP] Connecting to Qdrant di {QDRANT_PATH}...")
    qdrant_client = QdrantClient(path=QDRANT_PATH)

    print("[STARTUP] Loading Logistic Regression Reranker...")
    try:
        with open("lr_reranker.pkl", "rb") as f:
            lr_clf, label_encoder = pickle.load(f)
    except Exception:
        print("[STARTUP] lr_reranker.pkl tidak ditemukan. Training ulang dari Qdrant...")
        db_features = []
        db_labels = []
        offset = None
        while True:
            records, next_page = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=500,
                offset=offset,
                with_vectors=["cbir"],
                with_payload=True
            )
            for r in records:
                if r.vector and "cbir" in r.vector:
                    db_features.append(r.vector["cbir"])
                    db_labels.append(r.payload.get("class_name", "Unknown"))
                    
            if next_page is None:
                break
            offset = next_page
            
        if len(db_features) > 0:
            db_features = np.array(db_features, dtype=np.float32)
            db_labels = np.array(db_labels)
            
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(db_labels)
            
            lr_clf = LogisticRegression(max_iter=1000, class_weight='balanced')
            lr_clf.fit(db_features, y_encoded)
            
            with open("lr_reranker.pkl", "wb") as f:
                pickle.dump((lr_clf, label_encoder), f)
                
            print(f"[STARTUP] Logistic Regression reranker trained on {len(db_features)} items and saved.")
        else:
            print("[WARNING] Qdrant kosong, tidak dapat men-training Logistic Regression.")
            lr_clf = None
            label_encoder = None

    print("[STARTUP] Loading E5 Model & SPLADE for Hybrid Text Search...")
    e5_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    print("[STARTUP] All systems ready!")

# =============================================================================
# HELPER FUNCTIONS (Feature Extraction & Segmentation)
# =============================================================================
def get_mask_fallback(img):
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        mask_bin = (alpha > 10).astype(np.uint8) * 255
        bgr = img[:, :, :3]
        white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
        alpha_float = alpha.astype(np.float32) / 255.0
        for c in range(3):
            white_bg[:,:,c] = (alpha_float * bgr[:,:,c] + (1 - alpha_float) * white_bg[:,:,c])
        img_rgb = cv2.cvtColor(white_bg, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask_inv = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        mask_bin = cv2.bitwise_not(mask_inv)
    return img_rgb, mask_bin

def refine_bottle_mask(image_rgb, mask_pred):
    if mask_pred is None or np.sum(mask_pred) == 0:
        return mask_pred

    bin_mask = (mask_pred > 127).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    if num_labels <= 1:
        return bin_mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    main_mask = (labels == largest_label).astype(np.uint8) * 255

    contours, _ = cv2.findContours(main_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return main_mask
    main_cnt = max(contours, key=cv2.contourArea)
    filled_mask = np.zeros_like(main_mask)
    cv2.drawContours(filled_mask, [main_cnt], -1, 255, thickness=-1)

    ys, xs = np.where(filled_mask > 0)
    if len(ys) == 0:
        return filled_mask
    y_min, y_max = ys.min(), ys.max()
    h_bottle = y_max - y_min

    y_split = y_min + int(h_bottle * 0.35)
    if y_split < y_max:
        body_mask = filled_mask[y_split:y_max+1, :].copy()
        cnts, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            body_cnt = max(cnts, key=cv2.contourArea)
            hull = cv2.convexHull(body_cnt)
            cv2.drawContours(body_mask, [hull], -1, 255, thickness=-1)
            filled_mask[y_split:y_max+1, :] = body_mask

    h, w = filled_mask.shape
    left_profile = np.full(h, w, dtype=np.int32)
    right_profile = np.full(h, -1, dtype=np.int32)

    ys, xs = np.where(filled_mask > 0)
    for y in range(y_min, y_max + 1):
        row_xs = xs[ys == y]
        if len(row_xs) > 0:
            left_profile[y] = row_xs.min()
            right_profile[y] = row_xs.max()

    def smooth_profile(profile_segment, kernel_size=31):
        pad = kernel_size // 2
        padded = np.pad(profile_segment, (pad, pad), mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size)
        return np.median(windows, axis=1).astype(np.int32)

    l_seg = left_profile[y_min:y_max+1]
    r_seg = right_profile[y_min:y_max+1]

    k_size = max(11, min(41, h_bottle // 6))
    if k_size % 2 == 0: k_size += 1

    l_smooth = smooth_profile(l_seg, kernel_size=k_size)
    r_smooth = smooth_profile(r_seg, kernel_size=k_size)

    smooth_mask = np.zeros_like(filled_mask)
    for i, y in enumerate(range(y_min, y_max + 1)):
        lx = max(0, l_smooth[i])
        rx = min(w - 1, r_smooth[i])
        if lx <= rx:
            smooth_mask[y, lx:rx+1] = 255

    final_mask = cv2.GaussianBlur(smooth_mask, (7, 7), 2.0)
    return final_mask

def run_yolo_segmentation(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    results = yolo_model(img_bgr, verbose=False)
    
    if not results or len(results) == 0: 
        return None, False, 0.0
        
    r = results[0]
    if r.masks is None or r.boxes is None: 
        return None, False, 0.0
        
    best_mask = None
    best_score = 0.0
    h_orig, w_orig = img_rgb.shape[:2]
    
    for i, cls_idx in enumerate(r.boxes.cls):
        class_name = yolo_model.names[int(cls_idx)]
        score = float(r.boxes.conf[i])
        if class_name in ["bottle", "cup", "wine glass"] and score > best_score:
            best_score = score
            raw_mask = r.masks.data[i].cpu().numpy()
            best_mask = cv2.resize(raw_mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            
    if best_mask is None or best_score < 0.2: 
        return None, False, best_score
        
    best_mask_bin = (best_mask >= 0.32).astype(np.uint8) * 255
    refined_mask = refine_bottle_mask(img_rgb, best_mask_bin)
    best_mask_bin = (refined_mask > 127).astype(np.uint8) * 255
    
    return best_mask_bin, True, best_score

def extract_shape_features(mask_bin):
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return np.zeros(9, dtype=np.float32)
        
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

def pad_square(arr):
    h, w = arr.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side, 3), dtype=arr.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0:y0+h, x0:x0+w] = arr
    return canvas

@torch.no_grad()
def extract_cbir_vector(crop_rgb, crop_mask):
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
def extract_tbir_vector(text_query):
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



class MockPoint:
    def __init__(self, _id, score, payload):
        self.id = _id
        self.score = score
        self.payload = payload

# =============================================================================
# API SCHEMAS & ENDPOINTS
# =============================================================================
class SearchResult(BaseModel):
    id: Any
    score: float
    product_image_url: str
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    message: str
    segmentation_base64: Optional[str] = None
    results: List[SearchResult]

def image_to_base64(img_arr: np.ndarray) -> str:
    """Konversi numpy array BGR/RGB ke Base64 JPEG string"""
    if img_arr is None:
        return None
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def format_api_results(res_list) -> List[SearchResult]:
    api_results = []
    for r in res_list:
        payload = r.payload
        img_url = payload.get("ProductImageUrl", "")
        if not img_url:
            img_url = payload.get("image_path", "")
        
        api_results.append(SearchResult(
            id=r.id,
            score=r.score,
            product_image_url=img_url,
            metadata=payload
        ))
    return api_results


@app.post("/search/image", response_model=SearchResponse)
async def api_search_image(
    image: UploadFile = File(...)
):
    """
    Endpoint untuk mencari botol menggunakan gambar (CBIR).
    Menggunakan pipeline: YOLO Seg -> DINOv2 + Shape Feature -> Qdrant (cosine) -> XGBoost Reranker.
    """
    try:
        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        input_img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if input_img_bgr is None:
            raise HTTPException(status_code=400, detail="Gambar tidak valid atau corrupt.")
            
        image_rgb = cv2.cvtColor(input_img_bgr, cv2.COLOR_BGR2RGB)
        
        # Segmentasi
        mask_bin, is_yolo_success, score = run_yolo_segmentation(image_rgb)
        seg_display = None
        
        if not is_yolo_success:
            img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            image_rgb, mask_bin = get_mask_fallback(img_bgr)
            seg_display = image_rgb.copy()
        else:
            masked_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_bin)
            ys, xs = np.where(mask_bin > 0)
            y_min, y_max, x_min, x_max = ys.min(), ys.max(), xs.min(), xs.max()
            seg_display = masked_rgb[max(0, y_min-10):y_max+10, max(0, x_min-10):x_max+10]
            
            crop_rgb = masked_rgb[y_min:y_max+1, x_min:x_max+1]
            crop_mask = mask_bin[y_min:y_max+1, x_min:x_max+1]
        
        # Ekstrak CBIR Vector
        cbir_vec = extract_cbir_vector(crop_rgb, crop_mask)
        
        # Query ke Qdrant
        res_qdrant = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=cbir_vec.tolist(),
            using="cbir",
            limit=50,
            query_filter=None,
            with_payload=True
        ).points
            
        res = []
        if res_qdrant:
            if lr_clf is not None and label_encoder is not None:
                proba = lr_clf.predict_proba(cbir_vec.reshape(1, -1))[0]
                class_to_prob = {str(c): float(p) for c, p in zip(label_encoder.classes_, proba)}
                
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
            else:
                res = res_qdrant[:FINAL_K]

        if not res:
            return SearchResponse(
                message="⚠️ Tidak ada botol yang cocok di database (CBIR).",
                segmentation_base64=image_to_base64(seg_display) if seg_display is not None else None,
                results=[]
            )

        api_results = format_api_results(res)
        return SearchResponse(
            message=f"✅ Pencarian Gambar Selesai! (Menampilkan {len(api_results)} hasil)",
            segmentation_base64=image_to_base64(seg_display) if seg_display is not None else None,
            results=api_results
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Terjadi Kesalahan (CBIR): {str(e)}")


@app.post("/search/text", response_model=SearchResponse)
async def api_search_text(
    text_query: str = Form(...)
):
    """
    Endpoint untuk mencari botol menggunakan teks (TBIR).
    Menggunakan hybrid Ultimate Fusion: SigLIP (Visual Text) + BM25 + E5 -> RRF.
    """
    if not text_query.strip():
        raise HTTPException(status_code=400, detail="Deskripsi teks tidak boleh kosong.")
        
    try:
        tbir_vec = extract_tbir_vector(text_query)
        e5_vec = list(e5_model.embed([f"query: {text_query}"]))[0]
        splade_res = list(sparse_model.embed([text_query]))[0]

        prefetch_tbir = qmodels.Prefetch(
            query=tbir_vec.tolist(),
            using="tbir",
            limit=FINAL_K * 2
        )
        prefetch_e5 = qmodels.Prefetch(
            query=e5_vec.tolist(),
            using="e5_text",
            limit=FINAL_K * 2
        )
        prefetch_splade = qmodels.Prefetch(
            query=qmodels.SparseVector(
                indices=splade_res.indices.tolist(),
                values=splade_res.values.tolist()
            ),
            using="splade_text",
            limit=FINAL_K * 2
        )
        
        # Eksekusi Native Qdrant Fusion
        res = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[prefetch_tbir, prefetch_e5, prefetch_splade],
            query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
            limit=FINAL_K,
            with_payload=True
        ).points

        if not res:
            return SearchResponse(
                message="⚠️ Tidak ada botol yang cocok di database (TBIR).",
                results=[]
            )

        api_results = format_api_results(res)

        return SearchResponse(
            message=f"✅ Pencarian Teks Selesai! (Menampilkan {len(api_results)} hasil)",
            results=api_results
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Terjadi Kesalahan (TBIR): {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
