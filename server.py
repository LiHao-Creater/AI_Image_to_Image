from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from dinov2_numpy import Dinov2Numpy  # 你项目里的 numpy DINOv2

# ======================
# Config (按你的目录结构)
# ======================
ROOT = Path(__file__).resolve().parent

PROCESSED_JPG_DIR = ROOT / "processed_jpg"         # 用于前端展示
FEATURES_PATH = ROOT / "gallery_features.npy"      # (N,768) float16/float32, 建议已 L2 normalize
META_PATH = ROOT / "gallery_meta.csv"              # 包含 processed_jpg/caption/valid 等
WEIGHTS_PATH = ROOT / "vit-dinov2-base.npz"         # 权重

TOPK = 50

# 你想要更快：可把特征转 float32 常驻内存（15k*768*4 ≈ 46MB）
CACHE_FEATURES_FLOAT32 = True

# ======================
# Preprocess (与提特征保持一致：short side -> 224 + center crop 224 + ImageNet norm)
# ======================
IMGNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMGNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_pil_to_tensor(im: Image.Image) -> np.ndarray:
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    if min(w, h) <= 0:
        raise ValueError("invalid image size")

    scale = 224 / min(w, h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    im = im.resize((nw, nh), Image.BICUBIC)

    left = (nw - 224) // 2
    top = (nh - 224) // 2
    im = im.crop((left, top, left + 224, top + 224))

    arr = np.asarray(im, dtype=np.float32) / 255.0  # (224,224,3)
    arr = (arr - IMGNET_MEAN) / IMGNET_STD
    arr = np.transpose(arr, (2, 0, 1))  # (3,224,224)
    return arr[None, ...]  # (1,3,224,224)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


# ======================
# Load gallery + model at startup
# ======================
def must_exist(p: Path, hint: str):
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}\nHint: {hint}")


must_exist(PROCESSED_JPG_DIR, "请确认 processed_jpg/ 存在（展示图片目录）")
must_exist(FEATURES_PATH, "请先生成 gallery_features.npy（先跑 extract_features_no_download.py）")
must_exist(META_PATH, "请先生成 gallery_meta.csv（先跑 extract_features_no_download.py）")
must_exist(WEIGHTS_PATH, "请确认 vit-dinov2-base.npz 在当前目录")
must_exist(ROOT / "dinov2_numpy.py", "请确认 dinov2_numpy.py 在当前目录")

# 读 meta（只保留 valid=1 的行，避免坏图干扰）
meta_df = pd.read_csv(META_PATH)
if "valid" in meta_df.columns:
    meta_df = meta_df[meta_df["valid"].astype(int) == 1].copy()
meta_df.reset_index(drop=True, inplace=True)

# 需要的字段
if "processed_jpg" not in meta_df.columns:
    raise ValueError("gallery_meta.csv 必须包含 processed_jpg 列（如 000001.jpg）")
if "caption" not in meta_df.columns:
    meta_df["caption"] = ""

processed_files: List[str] = meta_df["processed_jpg"].astype(str).tolist()
captions: List[str] = meta_df["caption"].astype(str).tolist()

# 加载特征
gallery_feats = np.load(FEATURES_PATH, mmap_mode="r")
# 注意：如果你 meta 过滤了 valid，features 也要对应同样行。最佳做法是：
# 你的 gallery_meta.csv 本身就包含 row_index，与 features 对齐。
# 这里做一个更稳的对齐方式：如果有 row_index，按 row_index 取出 features。
if "row_index" in meta_df.columns:
    idx = meta_df["row_index"].astype(int).to_numpy()
    gallery_feats = gallery_feats[idx]

# 转 float32 常驻，提高 dot 速度
if CACHE_FEATURES_FLOAT32:
    gallery_matrix = np.asarray(gallery_feats, dtype=np.float32)
else:
    gallery_matrix = gallery_feats  # 可能是 float16 memmap

# 假设 features 已经 L2 normalize；为了稳，还是做一次 normalize（成本不大）
gallery_matrix = l2_normalize(np.asarray(gallery_matrix, dtype=np.float32))

# 加载模型
weights = np.load(WEIGHTS_PATH)
model = Dinov2Numpy(weights)

GALLERY_SIZE = int(gallery_matrix.shape[0])


# ======================
# Label inference (给你“这是什么”的判断 + probs)
# - Cat/Dog：TopK caption 里统计 cat/dog
# - WHIC：TopK caption 里取第一个词当“类名”做 Top-N 统计（简化版）
# ======================
def infer_catdog_from_topk(topk_caps: List[str]) -> Tuple[str, List[Dict]]:
    cat = 0
    dog = 0
    for c in topk_caps:
        s = c.lower()
        if "cat" in s:
            cat += 1
        if "dog" in s:
            dog += 1
    total = max(1, cat + dog)
    cat_p = cat / total
    dog_p = dog / total
    pred = "cat" if cat_p >= dog_p else "dog"
    probs = [{"label": "cat", "prob": float(cat_p)}, {"label": "dog", "prob": float(dog_p)}]
    return pred, probs


def infer_whic_from_topk(topk_caps: List[str], topn: int = 5) -> Tuple[str, List[Dict]]:
    # 简化：取 caption 第一个 token 做“类名”
    from collections import Counter

    def first_token(c: str) -> str:
        s = c.strip().lower()
        if not s:
            return "unknown"
        # 取第一个英文/数字 token
        import re
        m = re.match(r"([a-z0-9_+-]+)", s)
        return m.group(1) if m else "unknown"

    toks = [first_token(c) for c in topk_caps]
    cnt = Counter(toks)
    most = cnt.most_common(topn)
    total = sum(v for _, v in most) or 1
    probs = [{"label": k, "prob": float(v / total)} for k, v in most]
    pred = most[0][0] if most else "unknown"
    return pred, probs


# ======================
# FastAPI app
# ======================
app = FastAPI(title="ViT Retrieval Backend", version="1.0")

# 允许前端跨域（你说不用考虑安全）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态资源：让前端能访问图片
# 访问路径： http://127.0.0.1:8000/static/000001.jpg
app.mount("/static", StaticFiles(directory=str(PROCESSED_JPG_DIR)), name="static")


@app.get("/health")
def health():
    return {"ok": True, "gallery_size": GALLERY_SIZE}


def topk_cosine(query_vec: np.ndarray, k: int = TOPK) -> Tuple[np.ndarray, np.ndarray]:
    """
    query_vec: (1,768) float32 已归一化
    return (topk_scores, topk_indices) 按降序
    """
    # gallery_matrix: (N,768) 已归一化
    # cosine = dot
    sims = (gallery_matrix @ query_vec[0].astype(np.float32)).astype(np.float32)  # (N,)
    if k >= sims.shape[0]:
        idx = np.argsort(-sims)
        return sims[idx], idx
    # argpartition 更快
    part = np.argpartition(-sims, k)[:k]
    part = part[np.argsort(-sims[part])]
    return sims[part], part


@app.post("/api/retrieve")
async def api_retrieve(
    request: Request,
    file: UploadFile = File(...),
    source: str = Form("catdog"),  # "catdog" or "whic"
):
    t0 = time.time()
    req_id = uuid.uuid4().hex[:10]

    try:
        img_bytes = await file.read()
        im = Image.open(io.BytesIO(img_bytes))  # type: ignore
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"invalid image: {e}"})

    # 预处理
    try:
        x = preprocess_pil_to_tensor(im)  # (1,3,224,224)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"preprocess failed: {e}"})

    # 提特征
    feat = model(x.astype(np.float32))  # (1,768) float32
    feat = l2_normalize(feat.astype(np.float32))

    # TopK 检索
    scores, indices = topk_cosine(feat, k=TOPK)

    # 组装 topk 列表（image_url 用静态资源路径）
    base = str(request.base_url).rstrip("/")  # e.g. http://127.0.0.1:8000
    topk = []
    topk_caps = []

    for rank, (s, idx) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
        jpg_name = processed_files[idx]
        cap = captions[idx] if idx < len(captions) else ""
        topk_caps.append(cap)
        topk.append(
            {
                "rank": rank,
                "image_url": f"{base}/static/{jpg_name}",
                "caption": cap,
                "cosine": float(s),
                # score 你可以定义为 0-100 更直观
                "score": float(round(s * 100.0, 3)),
            }
        )

    # “这是什么”的判断 + probs
    src = source.lower().strip()
    if src == "catdog":
        pred, probs = infer_catdog_from_topk(topk_caps)
    else:
        pred, probs = infer_whic_from_topk(topk_caps, topn=5)

    elapsed_ms = int((time.time() - t0) * 1000)

    return {
        "source": "catdog" if src == "catdog" else "whic",
        "predicted_label": pred,
        "probs": probs,
        "gallery_size": GALLERY_SIZE,
        "elapsed_ms": elapsed_ms,
        "request_id": req_id,
        "topk": topk,
    }


# ---- 兼容：缺 io 的 import
import io  # 放最后也行
