"""图像检索演示 - 使用 demo 数据 (小规模)"""
import os
import numpy as np
from dinov2_numpy import Dinov2Numpy
from preprocess_image import resize_short_side

def l2_normalize(x, axis=-1, eps=1e-12):
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (denom + eps)

def _pick_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

print("=" * 60)
print("图像检索演示")
print("=" * 60)

weights_path = _pick_path(["vit-dinov2-base.npz", "./vit-dinov2-base.npz"])
if weights_path is None:
    raise FileNotFoundError("Cannot find vit-dinov2-base.npz")
weights = np.load(weights_path)
vit = Dinov2Numpy(weights)

cat_img = _pick_path(["./demo_data/cat.jpg", "cat.jpg", "./cat.jpg"])
dog_img = _pick_path(["./demo_data/dog.jpg", "dog.jpg", "./dog.jpg"])
if cat_img is None or dog_img is None:
    raise FileNotFoundError("Cannot find demo images (expected ./demo_data/cat.jpg and ./demo_data/dog.jpg).")

print("\n1. 构建图像库...")
gallery_images = [cat_img, dog_img]
gallery_features = []

for img_path in gallery_images:
    pixel_values = resize_short_side(img_path, target_size=224, patch_size=14)
    feat = vit(pixel_values)              # (1, 768)
    gallery_features.append(feat[0])      # -> (768,)
    print(f"   提取特征: {img_path}")

gallery_features = np.stack(gallery_features, axis=0).astype(np.float32)  # (N, 768)
print(f"   图库大小: {gallery_features.shape[0]} 张图像")
print(f"   特征形状: {gallery_features.shape}")

print("\n2. 查询图像...")
query_path = cat_img
query_pixel = resize_short_side(query_path, target_size=224, patch_size=14)
query_feat = vit(query_pixel)[0].astype(np.float32)  # (768,)
print(f"   查询: {query_path}")

print("\n3. 计算相似度（余弦相似度）...")
gallery_n = l2_normalize(gallery_features, axis=1)
query_n = l2_normalize(query_feat, axis=0)
similarities = gallery_n @ query_n  # (N,)

top_k = min(2, len(gallery_images))
top_indices = np.argsort(similarities)[::-1][:top_k]

print(f"\n4. Top-{top_k} 检索结果（cosine）:")
for rank, idx in enumerate(top_indices, 1):
    print(f"   {rank}. {gallery_images[idx]}")
    print(f"      相似度: {similarities[idx]:.4f}")

print("\n5. 使用 L2 距离:")
distances = np.linalg.norm(gallery_features - query_feat[None, :], axis=1)
top_indices_l2 = np.argsort(distances)[:top_k]

print(f"   Top-{top_k} 检索结果（L2）:")
for rank, idx in enumerate(top_indices_l2, 1):
    print(f"   {rank}. {gallery_images[idx]}")
    print(f"      距离: {distances[idx]:.4f}")

print("\n✓ 检索完成！")
print("\n" + "=" * 60)
print("扩展到大规模数据集（按 readme 要求）:")
print("1. 从 data.csv 下载 10,000+ 图像")
print("2. 用 resize_short_side 预处理所有图像（短边=224，padding 到 14 倍数）")
print("3. 批量提取并保存所有特征到 .npy（建议 float32 + 分块保存）")
print("4. 用户上传图像 -> 预处理 -> 提取特征 -> 计算相似度 -> 返回 Top-10")
print("=" * 60)
