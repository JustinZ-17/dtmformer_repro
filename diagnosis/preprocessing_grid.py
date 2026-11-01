# ============================================================
# 预处理诊断：一次跑完，直接告诉你模型要吃什么样的输入
# ============================================================
import os, sys, re, glob, collections
import numpy as np, cv2, torch, h5py

# ---------- 0) 保证模型和依赖都在 ----------
REPO = "/kaggle/working/DTMFormer"
sys.path.insert(0, REPO)
os.chdir(REPO)

import types
if "visualizer" not in sys.modules:
    _vis = types.ModuleType("visualizer")
    def _get_local(*a, **k):
        if len(a) == 1 and callable(a[0]):
            return a[0]
        return lambda f: f
    _vis.get_local = _get_local
    sys.modules["visualizer"] = _vis
    print("[补丁] visualizer 占位模块已就位")

device = "cuda"
if "model" not in globals():
    from models import SETR
    CKPT = globals().get("CKPT_ABS", os.path.join(REPO, "checkpoints_ACDC/Setr_DTMFormer.pth"))
    model = SETR.Setr_DTMFormer(classes=4).to(device)
    model.load_state_dict(torch.serialization.load(CKPT, map_location=device, weights_only=False))
    print("[模型] 重新加载完成")
else:
    print("[模型] 复用当前 kernel 里的 model")
model.eval()

# ---------- 1) 选几个完整 frame 整帧加载 ----------
paths = sorted(glob.glob("/kaggle/input/**/*_slice_*.h5", recursive=True))
assert paths, "没找到 h5，检查右侧 Input 是否挂着 acdc-dataset"

groups = collections.defaultdict(list)
for p in paths:
    m = re.match(r"patient(\d+)_(frame\d+)_slice_(\d+)", os.path.basename(p)[:-3])
    if m:
        groups[(int(m.group(1)), m.group(2))].append((int(m.group(3)), p))

frames = []
for key, lst in sorted(groups.items()):
    lst.sort()
    with h5py.File(lst[len(lst) // 2][1], "r") as f:
        r = float((f["label"][:] > 0).mean())
    if 0.03 < r < 0.45:
        imgs, labs = [], []
        for sid, p in lst:
            with h5py.File(p, "r") as f:
                imgs.append(f["image"][:].astype(np.float32))
                labs.append(f["label"][:].astype(np.uint8))
        frames.append((key, np.stack(imgs), np.stack(labs)))
    if len(frames) >= 6:
        break

n_slice = sum(f[1].shape[0] for f in frames)
print("选中 frame:", ["patient%03d_%s" % (k[0], k[1]) for k, _, _ in frames])
print("切片总数:", n_slice, "| 标签取值:", np.unique(frames[0][2]))

# ---------- 2) 先看一眼：图像和标签配准对不对 ----------
_, v0, l0 = frames[0]
img_mid, lab_mid = v0[len(v0) // 2], l0[len(l0) // 2]
inb = float(img_mid[lab_mid > 0].mean())
outb = float(img_mid[lab_mid == 0].mean())
print("\n[配准检查] 标签内平均亮度 = %.4f | 标签外平均亮度 = %.4f | 倍数 = %.2f"
      % (inb, outb, inb / max(outb, 1e-6)))
print("          倍数应明显大于 1，若接近 1 说明图像和标签没对齐")

# ---------- 3) 两套待选方案：亮度 x 几何 ----------
def intensity(vol, how):
    if how == "raw":
        return np.clip(vol * 255, 0, 255).astype(np.uint8)
    if how == "volmax":
        v = np.clip(vol / max(float(vol.max()), 1e-6), 0, 1)
        return (v * 255).astype(np.uint8)
    out = np.empty(vol.shape, np.uint8)
    for i in range(len(vol)):
        x = vol[i]
        if how == "minmax":
            v = (x - x.min()) / max(x.max() - x.min(), 1e-6)
        else:
            lo, hi = np.percentile(x, 1), np.percentile(x, 99)
            v = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        out[i] = (v * 255).astype(np.uint8)
    return out

def to256(im, geo, zoom=1.0, nearest=False):
    S = 256
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    h, w = im.shape
    if geo == "stretch":
        return cv2.resize(im, (S, S), interpolation=interp)
    if geo == "pad":
        s = zoom * S / max(h, w)
        nh, nw = max(int(round(h * s)), 1), max(int(round(w * s)), 1)
        r = cv2.resize(im, (nw, nh), interpolation=interp)
        if nh > S or nw > S:                    # 内容比画布大 -> 中心裁回来，别让补边变负数
            y0 = max((nh - S) // 2, 0)
            x0 = max((nw - S) // 2, 0)
            return r[y0:y0 + S, x0:x0 + S]
        t, b = (S - nh) // 2, S - nh - (S - nh) // 2
        l, rt = (S - nw) // 2, S - nw - (S - nw) // 2
        return cv2.copyMakeBorder(r, t, b, l, rt, cv2.BORDER_CONSTANT, value=0)
    if geo == "crop":
        side = min(max(int(min(h, w) / zoom), 1), h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        return cv2.resize(im[y0:y0 + side, x0:x0 + side], (S, S), interpolation=interp)
    raise ValueError(geo)

@torch.no_grad()
def predict(x8):
    t = torch.from_numpy(x8[None, None].astype(np.float32) / 255.0).to(device)
    out, _ = model(t)
    return out.softmax(1).argmax(1)[0].cpu().numpy().astype(np.uint8)

def class_dice(p, g):
    ds = {}
    for c in (1, 2, 3):
        a, b = (p == c), (g == c)
        if a.sum() == 0 and b.sum() == 0:
            ds[c] = float("nan")
        elif a.sum() == 0 or b.sum() == 0:
            ds[c] = 0.0
        else:
            ds[c] = 2 * (a & b).sum() / (a.sum() + b.sum())
    return ds

# ---------- 4) 第一轮：亮度 x 几何 网格搜索 ----------
print("\n" + "=" * 60)
print("%-10s%-10s%-10s%s" % ("亮度方案", "几何方式", "平均Dice", "备注"))
print("=" * 60)
grid = {}
for how in ["raw", "volmax", "minmax", "perc"]:
    for geo in ["stretch", "pad", "crop"]:
        alld = {}
        for c in (1, 2, 3):
            alld[c] = []
        for key, vol, lab in frames:
            im8 = intensity(vol, how)
            for i in range(len(im8)):
                if lab[i].max() == 0:
                    continue
                x = to256(im8[i], geo)
                y = to256(lab[i], geo, nearest=True)
                for c, d in class_dice(predict(x), y).items():
                    if not np.isnan(d):
                        alld[c].append(d)
        m = float(np.mean([np.mean(v) for v in alld.values() if v]))
        grid[(how, geo)] = (m, alld)
        mark = "   <== 当前方案" if (how == "perc" and geo == "stretch") else ""
        print("%-10s%-10s%-10.4f%s" % (how, geo, m, mark))

best = max(grid, key=lambda k: grid[k][0])
print("\n第一轮最优: 亮度=%s  几何=%s  Dice=%.4f" % (best[0], best[1], grid[best][0]))

# ---------- 5) 第二轮：内容缩放搜索 ----------
best_how = best[0]
best_geo = best[1]
best_zoom = 1.0
print("\n" + "=" * 60)
print("第二轮：内容缩放搜索（固定 亮度=%s 几何=%s）" % (best_how, best_geo))
print("=" * 60)
for geo in ["pad", "crop"]:
    for z in [0.80, 0.90, 0.95, 1.05, 1.15, 1.30]:
        vals = []
        for key, vol, lab in frames:
            im8 = intensity(vol, best_how)
            for i in range(len(im8)):
                if lab[i].max() == 0:
                    continue
                x = to256(im8[i], geo, zoom=z)
                y = to256(lab[i], geo, zoom=z, nearest=True)
                vals.append(np.nanmean(list(class_dice(predict(x), y).values())))
        m = float(np.mean(vals))
        flag = ""
        if m > grid[best][0]:
            grid[(best_how, geo)] = (m, {})
            best_geo, best_zoom = geo, z
            flag = "   <== 刷新最优"
        print("  %-6s zoom=%.2f  Dice=%.4f%s" % (geo, z, m, flag))

print("\n" + "#" * 60)
print("最终推荐组合：亮度=%s  几何=%s  zoom=%.2f" % (best_how, best_geo, best_zoom))
print("#" * 60)

# ---------- 6) 在最优组合下看每个类 + 混淆矩阵 + 出图 ----------
NAMES = {1: "RV 右心室", 2: "Myo 心肌", 3: "LV 左心室"}
per = {1: [], 2: [], 3: []}
conf = np.zeros((3, 3), dtype=np.int64)
slices_show = []

for key, vol, lab in frames:
    im8 = intensity(vol, best_how)
    for i in range(len(im8)):
        if lab[i].max() == 0:
            continue
        x = to256(im8[i], best_geo, zoom=best_zoom)
        y = to256(lab[i], best_geo, zoom=best_zoom, nearest=True)
        p = predict(x)
        for c, d in class_dice(p, y).items():
            if not np.isnan(d):
                per[c].append(d)
        for pc in (1, 2, 3):
            for gc in (1, 2, 3):
                conf[pc - 1, gc - 1] += int(((p == pc) & (y == gc)).sum())
        if key is frames[0][0] and len(slices_show) < 4 and i % max(len(im8) // 4, 1) == 0:
            slices_show.append((x, y, p))

print("\n各类 Dice：")
for c in (1, 2, 3):
    if per[c]:
        print("   %-10s %.4f" % (NAMES[c], float(np.mean(per[c]))))
print("   平均      %.4f" % float(np.mean([np.mean(per[c]) for c in per if per[c]])))

print("\n混淆矩阵（行=预测类，列=真实类，单位=像素数）：")
print("%-12s%14s%14s%14s" % ("", "真 RV", "真 Myo", "真 LV"))
for pc in (1, 2, 3):
    print("%-12s%14d%14d%14d" % ("pred " + NAMES[pc][:3],
                                 conf[pc - 1, 0], conf[pc - 1, 1], conf[pc - 1, 2]))

try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(slices_show), 3, figsize=(11, 3.4 * len(slices_show)))
    for r, (x, y, p) in enumerate(slices_show):
        axes[r, 0].imshow(x, cmap="gray")
        axes[r, 0].set_title("输入图像")
        axes[r, 1].imshow(y, vmin=0, vmax=3)
        axes[r, 1].set_title("真实标签 GT")
        axes[r, 2].imshow(p, vmin=0, vmax=3)
        axes[r, 2].set_title("DTMFormer 预测")
        for c in range(3):
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig("/kaggle/working/diag_predictions.png", dpi=130)
    plt.show()
    print("\n图片已存到 /kaggle/working/diag_predictions.png")
except Exception as e:
    print("\n[提示] 画图失败：", e)
