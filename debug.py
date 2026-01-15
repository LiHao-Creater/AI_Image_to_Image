import os
import numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__))
DEMO_DIR = os.path.join(ROOT, "demo_data")

from dinov2_numpy import Dinov2Numpy
from preprocess_image import center_crop

def must_exist(path: str, hint: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到：{path}\n提示：{hint}")
    return path

weights_path = must_exist(
    os.path.join(ROOT, "vit-dinov2-base.npz"),
    "请确认 vit-dinov2-base.npz 与 debug_fixed.py 在同一目录"
)

cat_img = must_exist(
    os.path.join(DEMO_DIR, "cat.jpg"),
    "请确认 demo_data/cat.jpg 存在"
)

dog_img = must_exist(
    os.path.join(DEMO_DIR, "dog.jpg"),
    "请确认 demo_data/dog.jpg 存在"
)

# reference：优先 demo_data 下，其次根目录
ref_path = os.path.join(DEMO_DIR, "cat_dog_feature.npy")
if not os.path.exists(ref_path):
    ref_path = os.path.join(ROOT, "cat_dog_feature.npy")
ref_path = must_exist(
    ref_path,
    "请确认 cat_dog_feature.npy 在 demo_data/ 或项目根目录"
)

weights = np.load(weights_path)
vit = Dinov2Numpy(weights)

cat_pixel_values = center_crop(cat_img)
cat_feat = vit(cat_pixel_values)

dog_pixel_values = center_crop(dog_img)
dog_feat = vit(dog_pixel_values)

reference = np.load(ref_path)
if reference.shape[0] != 2:
    raise ValueError(f"Reference should contain 2 features, got shape {reference.shape}")

cat_feat_ref = reference[0].reshape(1, -1).astype(np.float32, copy=False)
dog_feat_ref = reference[1].reshape(1, -1).astype(np.float32, copy=False)

cat_diff = np.abs(cat_feat - cat_feat_ref).max()
dog_diff = np.abs(dog_feat - dog_feat_ref).max()
cat_rel_err = cat_diff / (np.abs(cat_feat_ref).max() + 1e-12)
dog_rel_err = dog_diff / (np.abs(dog_feat_ref).max() + 1e-12)

print(f"Cat feature max absolute difference: {cat_diff:.6e}")
print(f"Dog feature max absolute difference: {dog_diff:.6e}")
print(f"Cat feature max relative error: {cat_rel_err:.6e}")
print(f"Dog feature max relative error: {dog_rel_err:.6e}")

if cat_diff < 0.01 and dog_diff < 0.01:
    print("✓ Features match reference within tolerance!")
else:
    print("✗ Features differ from reference significantly.")
