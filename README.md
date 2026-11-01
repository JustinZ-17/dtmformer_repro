# DTMFormer 复现记录（ACDC 心脏 MRI 分割）

在 Kaggle T4 上复现 DTMFormer（AAAI 2024）在 ACDC 上的分割实验，跑通官方代码与官方权重，
并系统排查了实测指标与论文数字的差距。

官方代码：https://github.com/iam-nacl/DTMFormer
本仓库不含官方代码本体，只收录复现流程、排查脚本与排障记录；官方代码由 Notebook 运行时 clone。

> **本仓库没有复现出论文指标。** 实测平均 Dice 0.4930，论文报告 92.53%。
> 第 4 节记录了实测数字和三条排查线，第 5 节列出当前结论与尚未闭合的疑点。

## 1. 结果速览

| 项 | 论文 | 本仓库实测 |
| --- | --- | --- |
| GFLOPs | 9.8 | 9.7840 |
| 参数量 | 33.5M | 33.527M |
| 平均 Dice | 92.53% | 0.4930 |
| RV（右心室） | — | 0.2289 |
| Myo（心肌） | — | 0.5501 |
| LV（左心室） | — | 0.7001 |
| HD95 | — | 11.11 voxel |

结构类指标（GFLOPs / 参数量）与论文吻合到小数点后一位，说明模型实现、权重加载、
网络规模三项都对。指标差距出现在分割精度上。

评估协议：`test.py --eval_mode patient`，100 个病人中取最后 20 个作为测试集，62 秒跑完。
论文没有公开 patient 划分方式，本项目的划分与作者未必一致。

## 2. 环境

| | |
| --- | --- |
| 平台 | Kaggle Notebook，GPU T4，Internet On |
| Python / PyTorch | 3.13 / 2.x CUDA |
| 手动安装 | `einops`、`thop`、`hausdorff`、`medpy` |
| 不装 | `visualizer`（见第 3 节坑 ③） |
| 数据集 | Kaggle `anhoangvo/acdc-dataset`（h5，float32，值域 0~1，原始尺寸 256×216） |
| 权重 | `checkpoints_ACDC/Setr_DTMFormer.pth`，143,838,716 字节（≈137.2 MiB），fp32，**未做任何微调** |

四个手动装的包都是作者的隐藏依赖，README 里一个字没提：

- `einops` —— patch 与 token 之间的张量变形（`Rearrange`），不装模型第一步就崩
- `thop` —— 统计 GFLOPs 与参数量，第 1 节那两个数靠它
- `hausdorff` —— `test.py` 顶部直接 import，不装报 ModuleNotFoundError
- `medpy` —— HD95 计算；装不上时 Notebook 内有同公式的 scipy 实现兜底

### 2.1 权重怎么拿到的

仓库里的 `Setr_DTMFormer.pth` 是 **Git LFS 指针**，只有 134 字节。真实文件必须走 media 通道：

https://media.githubusercontent.com/media/iam-nacl/DTMFormer/main/checkpoints_ACDC/Setr_DTMFormer.pth

下载后比对文件大小是否等于 143838716，这是判断拿到真权重而不是指针或截断文件的唯一办法。

## 3. 踩坑记录

六处坑全部来自作者代码在新环境下的隐含假设。

| # | 现象 | 根因 | 解法 |
| --- | --- | --- | --- |
| ① | 权重加载失败 | clone 下来的是 134 字节 LFS 指针 | 走 media 通道下载真实文件（见 2.1） |
| ② | `UnpicklingError` | PyTorch ≥ 2.6 起 `torch.load` 默认 `weights_only=True` | 补丁置为 False |
| ③ | `cannot import name 'get_local' from 'visualizer'` | PyPI 上同名包不含该函数；且该函数在全仓库从未被调用，是死代码 | 不装该包，注入占位模块到 `sys.modules` |
| ④ | `EinopsError: axis of length 154 in chunks of 8` | pipeline 全程无 resize，`--imgsize 256` 只作用于随机增强，val 模式一次不缩放 | 生成 PNG 时统一 resize 到 256×256；图像 `INTER_LINEAR`，标签 `INTER_NEAREST` |
| ⑤ | 只做评估也报 `FileNotFoundError` | 代码同时依赖 `trainofficial.txt` 与 `testofficial.txt`，按文件名前 3 位聚合成 patient | 文件名写成 `pid:03d_frame_slice.png`，列表按病人排序 |
| ⑥ | `RecursionError` | Notebook 里 `_raw = torch.load` 存的是全局变量，重跑单元格时它被重新赋值成上一轮的补丁本身 | 从 `torch.serialization.load` 取原函数，并在定义时用默认参数绑死：`def _patched_load(*a, _real=_TRUE_LOAD, **kw)` |

坑 ⑥ 只在 Notebook 里出现：同样的写法存成 `.py` 跑几次都没事，放进单元格重跑就自调用。

## 4. 指标差距排查

实测症状是**召回低、特异度极高**（灵敏度 0.17 / 0.46 / 0.65，特异度 0.999）——
该分割的区域没分割出来。据此列了三个方向逐个排除。

### 4.1 候选一：预处理与作者不一致 —— 排除

先读源码确认过：图像读 PNG 后经 `F.to_tensor` 直接进模型，全程无 resize、无归一化；
`z_score` 默认 False，且 `train.py` 与 `test.py` 的构造参数完全一致，不存在训测不一致。

再做受控网格搜索（`diagnosis/preprocessing_grid.py`），4 种灰度归一化 × 3 种几何变换：

| 亮度方案 | stretch | pad | crop |
| --- | --- | --- | --- |
| raw（原样 ×255） | 0.6095 | 0.5877 | 0.6074 |
| volmax | 0.6107 | 0.5882 | 0.6078 |
| minmax | 0.6105 | 0.5887 | 0.6076 |
| perc（分位截断） | 0.6117 | 0.5907 | 0.6092 |

12 个组合极差 0.024（0.588～0.612），目标差距是 0.49 对 0.92。
各方案彼此打平说明模型对输入侧改动不敏感，瓶颈不在这里。

一个被数据推翻的预判：这批 h5 已存成 0~1 float32，原本认为最贴近作者的写法「原样 ×255」
应当最好，实测 `raw` 0.6095 反而低于 `perc` 0.6117。

> 网格搜索的绝对值（0.61）与正式评估（0.49）不可直接比较：诊断脚本跑的是 24 张切片子集，
> 用的是自己实现的 Dice。只有组内比较有意义。

### 4.2 候选二：图像与标签空间错位 —— 排除

配准自检：标签区域内平均亮度 0.5947，标签外 0.2012，比值 2.96。远大于 1，图像与标签是对齐的。

再做扰动搜索（`diagnosis/alignment_search.py`）：不动模型和图像，只对 GT 做平移 / 翻转 / 旋转。

| 变换 | Dice | 相对基准 |
| --- | --- | --- |
| 原样（基准） | **0.6186** | — |
| 左移 2px | 0.6185 | −0.0002 |
| 下移 2px | 0.5956 | −0.0231 |
| 左移 4px | 0.5723 | −0.0463 |
| 右移 8px | 0.3639 | −0.2547 |
| 左右翻转 | 0.2833 | −0.3353 |
| 旋转 180° | 0.2684 | −0.3502 |
| 左移 32px | 0.1158 | −0.5028 |
| 上移 32px | 0.0845 | −0.5341 |

24 项测试中基准即最高分，其余全为负，且衰减极陡（32px 已掉到 0.08）。
预测与 GT 在位置上已是最优，任何挪动只会更差。

### 4.3 候选三：checkpoint 本身 —— 当前结论

三条排查线全部阴性，剩下的解释是：官方放出的这份 checkpoint 达不到论文数字
（早停 / 未收敛，或论文用的是未公开的病人划分）。

0.49 与 0.92 的差距超出「划分不同」能解释的量级。

## 5. 尚未闭合的疑点

排查没有就此收尾，以下几条待下次上机验证：

1. **92.53% 是否对应这个权重。** 官方 README 未指明该数字对应哪个 `.pth`；
   `test.py` 里默认权重路径写的是 `/home/wzh/code/SETR-pytorch/...`，指向另一个仓库。
   需要查论文 Table 1 确认 SETR 行的独立成绩。
2. **checkpoint 体积比参数量多出约 9.28 MiB。** 33.527M 参数 ×4 字节 ≈ 134.1MB，
   文件实际 137.2 MiB，差额约 243 万个 float32。可能是 `thop` 只统计参与前向的模块，
   而文件里另有未被覆盖的分支。验证：`set(ckpt.keys()) - set(model.state_dict().keys())`。
3. **RV 接近整类漏检。** 可视化里有若干病人的 RV 完全没有预测出来（不是分割得歪，
   是该类别为空），而 Myo 与 LV 位置正确。这与 RV Dice 0.2289 吻合。
   下一跳假设是 attention collapse 导致预测偏向像素占比最大的类，尚无数据支撑。
4. **作者未开源 nii→PNG 转换脚本**，训练数据的长宽比如何处理无从得知。

## 6. 待办

- [ ] 查论文 Table 1，确认 SETR + DTMFormer 行的成绩（疑点 1）
- [ ] 对比 ckpt 与 state_dict 的 key 差异（疑点 2）
- [ ] dump 中间层 attention，检查各行分布是否趋同（疑点 3）
- [ ] 用 ACDC 官方数据跑 `train.py`，对比自训权重与官方权重

## 7. 文件

| 文件 | 内容 |
| --- | --- |
| `DTMFormer_Kaggle_复现.ipynb` | 完整流程：环境 → 权重 → 数据 → 评估 → 可视化 → 排查 |
| `diagnosis/preprocessing_grid.py` | 预处理网格搜索 + 配准自检 |
| `diagnosis/alignment_search.py` | 空间错位扰动搜索 |
| `requirements.txt` | 依赖清单 |

## 8. 参考

- DTMFormer: *Dynamic Token Merging for Boosting Transformer-Based Medical Image Segmentation* (AAAI 2024)，DOI `10.1609/aaai.v38i6.28394`
- 官方代码：https://github.com/iam-nacl/DTMFormer
- ACDC：https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
- Kaggle 数据集镜像：https://www.kaggle.com/datasets/anhoangvo/acdc-dataset
