# ============================================================
# 空间错位搜索 —— 独立于 notebook，可在 Kaggle 单独运行
# ------------------------------------------------------------
# 背景：跑通官方代码后 Dice 远低于论文。排查顺序是
#   ① 预处理是否和作者一致  ② 图像与标签是否空间错位  ③ 权重本身
# 本脚本负责第 ② 步。
#
# 思路：不动模型、不动图片，只对【真值 GT】做几何扰动。
#   - 若某个扰动让 Dice 明显变高 -> 说明数据在那個方向上错位了
#   - 若基准就是最高、其余全为负 -> 错位排除，转去查 ③
#
# 运行方式：在已跑通 notebook 的 Kaggle kernel 里直接粘_script 运行即可；
#          若 kernel 重启过，下面会自动重新加载官方权重。
# ============================================================
import os, sys, glob, types
import numpy as np
import cv2
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/kaggle/working/DTMFormer"
CKPT = os.path.join(REPO, "checkpoints_ACDC", "Setr_DTMFormer.pth")
SRC_GLOB = "/kaggle/input/**/*_slice_*.h5"
N_CASES, SIZE = 24, 256

sys.path.insert(0, REPO)

# ---------- 0) visualizer 占位 ----------
# 作者在 models/model_utils.py 里 `from visualizer import get_local`，
# 但全仓库从未调用它（死代码）。PyPI 上同名包里又没有这个函数，
# 因此在这里塞一个占位模块，既满足 import 又零副作用。
if "visualizer" not in sys.modules:
    vis = types.ModuleType("visualizer")
    def _get_local(*a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return lambda f: f
    vis.get_local = _get_local
    sys.modules["visualizer"] = vis

# ---------- 1) 模型（复用 kernel 里的 model，没有就重新加载） ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
if "model" not in globals():
    from models import SETR
    model = SETR.Setr_DTMFormer(classes=4).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False))
model.eval()
print(f"device = {device}")

# ---------- 2) 工具函数 ----------
def dice3(pred, gt):
    """三个前景类（RV/Myo/LV）的平均 Dice，背景类 0 不计入"""
    vals = []
    for c in (1, 2, 3):
        a, b = (pred == c), (gt == c)
        if a.sum() == 0 and b.sum() == 0:
            continue
        if a.sum() == 0 or b.sum() == 0:
            vals.append(0.0)
            continue
        vals.append(2.0 * (a & b).sum() / (a.sum() + b.sum()))
    return float(np.mean(vals)) if vals else float("nan")

@torch.no_grad()
def predict(x8):
    """x8: uint8 的 (256,256) 灰度图 -> (256,256) 预测类别图"""
    t = torch.from_numpy(x8[None, None].astype(np.float32) / 255.0).to(device)
    out, _ = model(t)
    return out.softmax(1).argmax(1)[0].cpu().numpy().astype(np.uint8)

def perc_u8(x):
    """1%/99% 分位拉伸：去掉 MRI 里极端的亮暗点"""
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return (np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)

def sq(im, nearest=False):
    """统一缩放到 256x256。
    注意：标签必须用 NEAREST —— 它存的是类别编号 0/1/2/3，
    用双线性会插值出 1.7 这种不存在的类别，真值直接毁掉。"""
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(im, (SIZE, SIZE), interpolation=interp)

# ---------- 3) 挑一批「有心脏、大小适中」的切片 ----------
all_h5 = sorted(glob.glob(SRC_GLOB, recursive=True))
assert all_h5, "找不到 *_slice_*.h5，检查数据集是否已挂载"

cases = []
for p in all_h5:
    with h5py.File(p, "r") as f:
        lab = f["label"][:]
    # 过滤掉：全背景的切片、以及心脏几乎占满画面的切片
    if lab.max() > 0 and 0.02 < float((lab > 0).mean()) < 0.5:
        cases.append(p)
    if len(cases) >= N_CASES:
        break
print("标定切片数:", len(cases))

X, PREDS, GTS = [], [], []
for p in cases:
    with h5py.File(p, "r") as f:
        img, lab = f["image"][:], f["label"][:]
    x = sq(perc_u8(img))
    X.append(x)
    GTS.append(sq(lab.astype(np.uint8), nearest=True))
    PREDS.append(predict(x))
print("预测完成")

# ---------- 4) 只对 GT 做几何扰动 ----------
def roll(a, dy, dx):
    return np.roll(a, (dy, dx), axis=(0, 1))

variants = {"原样(基准)": lambda m: m}
for d in (2, 4, 8, 16, 32):
    variants[f"GT下移{d}px"] = lambda m, d=d: roll(m,  d, 0)
    variants[f"GT上移{d}px"] = lambda m, d=d: roll(m, -d, 0)
    variants[f"GT右移{d}px"] = lambda m, d=d: roll(m, 0,  d)
    variants[f"GT左移{d}px"] = lambda m, d=d: roll(m, 0, -d)
variants["GT左右翻转"] = lambda m: m[:, ::-1]
variants["GT上下翻转"] = lambda m: m[::-1, :]
variants["GT旋转180"]  = lambda m: m[::-1, ::-1]

results = []
for name, fn in variants.items():
    scores = [dice3(pr, fn(gt)) for pr, gt in zip(PREDS, GTS)]
    results.append((name, float(np.nanmean(scores))))

baseline = dict(results)["原样(基准)"]
results.sort(key=lambda r: -r[1])

print()
print("对【真值】做几何扰动，看 Dice 会不会涨")
print(f"{'变换':<16}{'Dice':<10}{'相对基准'}")
print("-" * 44)
for name, val in results:
    print(f"{name:<16}{val:<10.4f}{val - baseline:+.4f}")

best_name, best_val = results[0]
print()
if best_name == "原样(基准)":
    print("=> 基准就是最优，任何移动都只会变差 -> 无空间错位")
else:
    print(f"=> 最优是 {best_name}（{best_val:.4f}），可能存在该方向的错位")

# ---------- 5) 出图定性 ----------
palette = np.array([[0, 0, 0], [255, 80, 80], [80, 200, 120], [80, 140, 255]], np.uint8)
show = min(3, len(X))
fig, axes = plt.subplots(show, 3, figsize=(11, 3.4 * show))
axes = np.atleast_2d(axes)
for r in range(show):
    base_rgb = np.dstack([X[r]] * 3)
    ov_gt, ov_pr = base_rgb.copy(), base_rgb.copy()
    m_gt, m_pr = GTS[r] > 0, PREDS[r] > 0
    ov_gt[m_gt] = palette[GTS[r][m_gt]]
    ov_pr[m_pr] = palette[PREDS[r][m_pr]]
    axes[r, 0].imshow(X[r], cmap="gray")
    axes[r, 0].set_title(os.path.basename(cases[r]), fontsize=9)
    axes[r, 1].imshow(ov_gt)
    axes[r, 1].set_title("真值 GT", fontsize=9)
    axes[r, 2].imshow(ov_pr)
    axes[r, 2].set_title("DTMFormer 预测", fontsize=9)
    for c in range(3):
        axes[r, c].axis("off")
plt.tight_layout()
out_png = "/kaggle/working/check_align.png"
plt.savefig(out_png, dpi=120)
print("图已保存:", out_png)
