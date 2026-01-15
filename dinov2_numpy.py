import numpy as np

# ============================================================
# Pure NumPy cubic-spline resize
# - Cubic B-spline basis
# - Prefilter: solve tridiagonal system (mode='nearest')
# - Sampling: weights from raw indices; indices clipped for nearest boundary
# This is designed to match the reference feature generation closely.
# ============================================================

def _bspline3(x: np.ndarray) -> np.ndarray:
    ax = np.abs(x).astype(np.float32)
    out = np.zeros_like(ax, dtype=np.float32)

    m1 = ax < 1.0
    m2 = (ax >= 1.0) & (ax < 2.0)

    out[m1] = (4.0 - 6.0 * ax[m1] ** 2 + 3.0 * ax[m1] ** 3) / 6.0
    t = 2.0 - ax[m2]
    out[m2] = (t ** 3) / 6.0
    return out

def _spline_coeffs_cubic_nearest_1d(s: np.ndarray) -> np.ndarray:
    """
    Cubic B-spline prefilter coefficients with boundary mode='nearest'.
    s: (n, m) float32
    returns: (n, m) float32
    """
    s = s.astype(np.float32, copy=False)
    n = s.shape[0]
    m = s.shape[1]

    # Tridiagonal system:
    # interior: (1/6)c[i-1] + (4/6)c[i] + (1/6)c[i+1] = s[i]
    # boundary (nearest): (5/6)c[0] + (1/6)c[1] = s[0]
    #                    (1/6)c[n-2] + (5/6)c[n-1] = s[n-1]
    a = np.full(n, 1.0 / 6.0, dtype=np.float32)
    b = np.full(n, 4.0 / 6.0, dtype=np.float32)
    c = np.full(n, 1.0 / 6.0, dtype=np.float32)
    a[0] = 0.0
    c[-1] = 0.0
    b[0] = 5.0 / 6.0
    b[-1] = 5.0 / 6.0

    # Thomas algorithm (vectorized over m)
    cp = np.empty(n, dtype=np.float32)
    dp = np.empty((n, m), dtype=np.float32)

    cp[0] = c[0] / b[0]
    dp[0] = s[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = (c[i] / denom) if i < n - 1 else 0.0
        dp[i] = (s[i] - a[i] * dp[i - 1]) / denom

    out = np.empty((n, m), dtype=np.float32)
    out[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        out[i] = dp[i] - cp[i] * out[i + 1]
    return out

def _spline_filter2d_nearest(x: np.ndarray) -> np.ndarray:
    """
    Separable cubic-spline prefilter on (H,W,C) with nearest boundary.
    Returns coefficient grid.
    """
    x = x.astype(np.float32, copy=False)
    H, W, C = x.shape

    tmp = _spline_coeffs_cubic_nearest_1d(x.reshape(H, W * C)).reshape(H, W, C)
    tmp_t = tmp.transpose(1, 0, 2).reshape(W, H * C)
    out = _spline_coeffs_cubic_nearest_1d(tmp_t).reshape(W, H, C).transpose(1, 0, 2)
    return out

def _zoom_spline3_nearest(x: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Resize (H,W,C)->(out_h,out_w,C) using cubic spline (order=3), mode='nearest'.
    """
    x = x.astype(np.float32, copy=False)
    H, W, C = x.shape
    if H == out_h and W == out_w:
        return x

    coeff = _spline_filter2d_nearest(x)

    # Coordinate mapping aligned to endpoints (common ndimage zoom behavior)
    if out_h == 1:
        iy = np.zeros((out_h,), dtype=np.float32)
    else:
        iy = np.arange(out_h, dtype=np.float32) * (H - 1) / (out_h - 1)
    y0 = np.floor(iy).astype(np.int32)
    raw_y = np.stack([y0 - 1, y0, y0 + 1, y0 + 2], axis=1)  # (out_h, 4)
    wy = _bspline3(iy[:, None] - raw_y.astype(np.float32))
    y_idx = np.clip(raw_y, 0, H - 1)
    tmp = np.sum(coeff[y_idx] * wy[:, :, None, None], axis=1)  # (out_h, W, C)

    if out_w == 1:
        ix = np.zeros((out_w,), dtype=np.float32)
    else:
        ix = np.arange(out_w, dtype=np.float32) * (W - 1) / (out_w - 1)
    x0 = np.floor(ix).astype(np.int32)
    raw_x = np.stack([x0 - 1, x0, x0 + 1, x0 + 2], axis=1)  # (out_w, 4)
    wx = _bspline3(ix[:, None] - raw_x.astype(np.float32))
    x_idx = np.clip(raw_x, 0, W - 1)
    out = np.sum(tmp[:, x_idx, :] * wx[None, :, :, None], axis=2)  # (out_h, out_w, C)

    return out.astype(np.float32, copy=False)

# -----------------------------
# Core ops
# -----------------------------
def gelu(x):
    x = x.astype(np.float32, copy=False)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(np.float32(2.0 / np.pi)) * (x + np.float32(0.044715) * np.power(x, 3))))

def softmax(x, axis=-1):
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)

class Embeddings:
    def __init__(self, weights):
        self.hidden_size = 768
        self.patch_size  = 14

        self.cls_token = weights["embeddings.cls_token"].astype(np.float32, copy=False)  # (1,1,D)
        self.position_embeddings = weights["embeddings.position_embeddings"].astype(np.float32, copy=False)  # (1,N+1,D)

        # Projection weights
        self.patch_embed_w = weights["embeddings.patch_embeddings.projection.weight"].reshape(768, -1).T.astype(np.float32, copy=False)
        self.patch_embed_b = weights["embeddings.patch_embeddings.projection.bias"].reshape(768, 1).T.astype(np.float32, copy=False)

    def pixel2patches(self, pixel_values):
        """
        (B,C,H,W) -> (B, h*w, C*ps*ps) in row-major patch order
        """
        B, C, H, W = pixel_values.shape
        ps = self.patch_size
        if H % ps != 0 or W % ps != 0:
            raise ValueError(f"Input H,W must be multiples of patch_size={ps}, got H={H}, W={W}")
        h = H // ps
        w = W // ps

        x = pixel_values.astype(np.float32, copy=False).reshape(B, C, h, ps, w, ps)
        x = x.transpose(0, 2, 4, 1, 3, 5)          # (B,h,w,C,ps,ps)
        patches = x.reshape(B, h * w, C * ps * ps) # (B, h*w, C*ps*ps)
        return patches

    def interpolate_pos_encoding(self, embeddings, height, width):
        B, Np1, D = embeddings.shape
        ps = self.patch_size
        new_h = height // ps
        new_w = width  // ps
        new_n = new_h * new_w

        if new_n + 1 != Np1:
            raise ValueError(
                f"Token mismatch: embeddings has {Np1} tokens, but H,W imply {new_n+1}. "
                f"Make sure H,W divisible by {ps}."
            )

        pos = self.position_embeddings  # (1, old_n+1, D)
        cls_pos = pos[:, :1, :]
        patch_pos = pos[:, 1:, :]
        old_n = patch_pos.shape[1]

        if old_n == new_n:
            return np.tile(pos, (B, 1, 1))

        old_size = int(np.sqrt(old_n))
        if old_size * old_size != old_n:
            raise ValueError(f"Expected square pos grid, got old_n={old_n}")

        patch_2d = patch_pos.reshape(old_size, old_size, D)
        resized = _zoom_spline3_nearest(patch_2d, new_h, new_w).reshape(1, new_n, D)

        pos_resized = np.concatenate([cls_pos, resized], axis=1)
        return np.tile(pos_resized, (B, 1, 1))

    def __call__(self, pixel_values):
        B, _, H, W = pixel_values.shape

        patch_values = self.pixel2patches(pixel_values)
        embeddings = patch_values @ self.patch_embed_w + self.patch_embed_b  # (B,N,D)

        cls_token = np.tile(self.cls_token, (B, 1, 1))  # (B,1,D)
        embeddings = np.concatenate([cls_token, embeddings], axis=1)  # (B,N+1,D)

        pos_embed = self.interpolate_pos_encoding(embeddings, H, W)
        return embeddings + pos_embed

class LayerNorm:
    def __init__(self, weight, bias, eps=1e-6):
        self.weight = weight.astype(np.float32, copy=False)
        self.bias   = bias.astype(np.float32, copy=False)
        self.eps    = np.float32(eps)

    def __call__(self, x):
        x = x.astype(np.float32, copy=False)
        mean = x.mean(-1, keepdims=True)
        var  = ((x - mean) ** 2).mean(-1, keepdims=True)
        x = (x - mean) / np.sqrt(var + self.eps)
        return x * self.weight + self.bias

class Linear:
    def __init__(self, weight, bias):
        self.weight = weight.astype(np.float32, copy=False)
        self.bias   = bias.astype(np.float32, copy=False)

    def __call__(self, x):
        x = x.astype(np.float32, copy=False)
        return x @ self.weight.T + self.bias

class MultiHeadAttention:
    def __init__(self, config, prefix, weights):
        self.num_heads = config['num_heads']
        self.head_dim = config['hidden_size'] // self.num_heads

        q_w = weights[f"{prefix}.attention.query.weight"]
        q_b = weights[f"{prefix}.attention.query.bias"]
        k_w = weights[f"{prefix}.attention.key.weight"]
        k_b = weights[f"{prefix}.attention.key.bias"]
        v_w = weights[f"{prefix}.attention.value.weight"]
        v_b = weights[f"{prefix}.attention.value.bias"]
        o_w = weights[f"{prefix}.output.dense.weight"]
        o_b = weights[f"{prefix}.output.dense.bias"]

        self.q_proj   = Linear(q_w, q_b)
        self.k_proj   = Linear(k_w, k_b)
        self.v_proj   = Linear(v_w, v_b)
        self.out_proj = Linear(o_w, o_b)

    def __call__(self, x):
        x = x.astype(np.float32, copy=False)
        B, N, D = x.shape
        H = self.num_heads
        Hd = self.head_dim
        if D != H * Hd:
            raise ValueError(f"hidden_size {D} != num_heads*head_dim {H}*{Hd}")

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(B, N, H, Hd).transpose(0, 2, 1, 3)  # (B,H,N,Hd)
        k = k.reshape(B, N, H, Hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, H, Hd).transpose(0, 2, 1, 3)

        att = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(np.float32(Hd))
        att = softmax(att, axis=-1)

        out = np.matmul(att, v)  # (B,H,N,Hd)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(out)

class MLP:
    def __init__(self, prefix, weights):
        w1 = weights[f"{prefix}.mlp.fc1.weight"]
        b1 = weights[f"{prefix}.mlp.fc1.bias"]
        w2 = weights[f"{prefix}.mlp.fc2.weight"]
        b2 = weights[f"{prefix}.mlp.fc2.bias"]

        self.fc1 = Linear(w1, b1)
        self.fc2 = Linear(w2, b2)

    def __call__(self, x):
        return self.fc2(gelu(self.fc1(x)))

class LayerScale:
    def __init__(self, weight):
        self.lambda1 = weight.astype(np.float32, copy=False)

    def __call__(self, x):
        return x * self.lambda1

class TransformerBlock:
    def __init__(self, config, idx, weights):
        prefix = f"encoder.layer.{idx}"

        self.norm1 = LayerNorm(weights[f"{prefix}.norm1.weight"], weights[f"{prefix}.norm1.bias"])
        self.scale1 = LayerScale(weights[f"{prefix}.layer_scale1.lambda1"])
        self.attn = MultiHeadAttention(config, f"{prefix}.attention", weights)

        self.norm2 = LayerNorm(weights[f"{prefix}.norm2.weight"], weights[f"{prefix}.norm2.bias"])
        self.scale2 = LayerScale(weights[f"{prefix}.layer_scale2.lambda1"])
        self.mlp = MLP(prefix, weights)

    def __call__(self, x):
        x = x + self.scale1(self.attn(self.norm1(x)))
        x = x + self.scale2(self.mlp(self.norm2(x)))
        return x

class Dinov2Numpy:
    def __init__(self, weights, config=None):
        self.config = config or {
            "hidden_size": 768,
            "num_heads": 12,
            "num_layers": 12,
        }

        self.embeddings = Embeddings(weights)
        self.blocks = [TransformerBlock(self.config, i, weights) for i in range(self.config["num_layers"])]
        self.norm = LayerNorm(weights["layernorm.weight"], weights["layernorm.bias"])

    def __call__(self, pixel_values):
        x = self.embeddings(pixel_values)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]
