#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
query_gallery.py

Query a built gallery (features + urls) and return Top-K similar images.

Usage:
  python query_gallery.py --gallery_dir gallery_out --query_url "http://..." --topk 10
  python query_gallery.py --gallery_dir gallery_out --query_path "demo_data/cat.jpg" --topk 10
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys
import time
from typing import List, Tuple

import numpy as np
from PIL import Image
import urllib.request

from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "identity",
    "Connection": "close",
}

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (denom + eps)

def load_gallery_parts(gallery_dir: str) -> List[Tuple[str, str, str]]:
    feat_files = sorted(glob.glob(os.path.join(gallery_dir, "gallery_features_part*.npy")))
    parts = []
    for fp in feat_files:
        base = os.path.basename(fp)
        tag = base.replace("gallery_features_", "").replace(".npy", "")
        urlp = os.path.join(gallery_dir, f"gallery_urls_{tag}.npy")
        capp = os.path.join(gallery_dir, f"gallery_captions_{tag}.npy")
        if not os.path.exists(urlp):
            raise FileNotFoundError(f"Missing urls file for {fp}: expected {urlp}")
        if not os.path.exists(capp):
            capp = ""
        parts.append((fp, urlp, capp))
    return parts

def download_with_retries(url: str, dst_path: str, timeout: int, retries: int, backoff: float) -> str:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    last_err = ""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=DEFAULT_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            with open(dst_path, "wb") as f:
                f.write(data)
            # validate
            with Image.open(dst_path) as im:
                im.convert("RGB")
            return dst_path
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            try:
                if os.path.exists(dst_path):
                    os.remove(dst_path)
            except Exception:
                pass
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
            else:
                raise RuntimeError(f"Failed to download query url: {last_err}")
    raise RuntimeError(f"Failed to download query url: {last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery_dir", default="gallery_out")
    ap.add_argument("--query_url", default="")
    ap.add_argument("--query_path", default="")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--cache_dir", default="cache_images")
    ap.add_argument("--timeout", type=int, default=12)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--backoff", type=float, default=0.8)
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    gallery_dir = os.path.join(root, args.gallery_dir) if not os.path.isabs(args.gallery_dir) else args.gallery_dir
    cache_dir = os.path.join(root, args.cache_dir) if not os.path.isabs(args.cache_dir) else args.cache_dir

    weights_path = os.path.join(root, "vit-dinov2-base.npz")
    if not os.path.exists(weights_path):
        print(f"[ERROR] Cannot find weights at: {weights_path}", file=sys.stderr)
        sys.exit(1)

    parts = load_gallery_parts(gallery_dir)
    if not parts:
        print(f"[ERROR] No gallery parts found in {gallery_dir}", file=sys.stderr)
        sys.exit(1)

    # Resolve query image to local path
    if args.query_path:
        qpath = args.query_path
        if not os.path.isabs(qpath):
            qpath = os.path.join(root, qpath)
        if not os.path.exists(qpath):
            raise FileNotFoundError(f"Query path not found: {qpath}")
    elif args.query_url:
        os.makedirs(cache_dir, exist_ok=True)
        fname = hashlib.sha1(args.query_url.encode("utf-8", errors="ignore")).hexdigest() + ".jpg"
        qpath = os.path.join(cache_dir, fname)
        if not os.path.exists(qpath):
            qpath = download_with_retries(args.query_url, qpath, timeout=args.timeout, retries=args.retries, backoff=args.backoff)
    else:
        raise ValueError("Provide --query_url or --query_path")

    # Embed query
    weights = np.load(weights_path)
    vit = Dinov2Numpy(weights)
    qpix = resize_short_side(qpath, target_size=224, patch_size=14)
    qfeat = vit(qpix)[0].astype(np.float32, copy=False)
    qfeat = l2_normalize(qfeat, axis=0)

    topk = max(1, args.topk)
    best_scores = np.full((topk,), -1e9, dtype=np.float32)
    best_urls = [""] * topk
    best_caps = [""] * topk

    for feat_path, url_path, cap_path in parts:
        feats = np.load(feat_path).astype(np.float32, copy=False)
        urls = np.load(url_path, allow_pickle=True)
        caps = np.load(cap_path, allow_pickle=True) if cap_path else np.array([""] * len(urls), dtype=object)

        feats_n = l2_normalize(feats, axis=1)
        scores = feats_n @ qfeat

        k = min(topk, len(scores))
        idx = np.argpartition(-scores, kth=k-1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        for i in idx:
            s = float(scores[i])
            j = int(np.argmin(best_scores))
            if s > best_scores[j]:
                best_scores[j] = s
                best_urls[j] = str(urls[i])
                best_caps[j] = str(caps[i]) if i < len(caps) else ""

    order = np.argsort(-best_scores)
    best_scores = best_scores[order]
    best_urls = [best_urls[i] for i in order]
    best_caps = [best_caps[i] for i in order]

    print("=" * 80)
    print("Top results (cosine):")
    for r, (u, c, s) in enumerate(zip(best_urls, best_caps, best_scores), 1):
        if not u:
            continue
        print(f"{r:02d}. score={s:.4f}")
        print(f"    url: {u}")
        if c:
            print(f"    caption: {c}")
    print("=" * 80)

if __name__ == "__main__":
    main()
