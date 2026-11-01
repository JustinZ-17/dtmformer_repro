# DTMFormer 复现：ACDC 心脏 MRI 分割

在 Kaggle T4 上复现 **DTMFormer**（AAAI 2024）在 ACDC 心脏 MRI 数据集上的分割实验，
包括环境搭建、官方权重恢复、数据转换、patient 级评估，以及指标差距的系统排查。

**官方代码**：https://github.com/iam-nacl/DTMFormer

**官方仓库与本仓库的关系**：本仓库**不含官方代码本体**，只收录我自己写的复现流程
（Kaggle Notebook）、排查脚本与完整的排障记录。官方代码在运行时由 Notebook 自动 clone。

> **重要说明**：本仓库目前**没有复现出论文报告的指标**（Dice 92.53%）。
> 第五、六节如实记录了实测数字与我排查过并**逐个排除**的三个方向。
> 已确认成立的部分（模型结构、评估协议、数据正确性）在第三节列出。

---

## 1. 目标与方法

### 1.1 DTMFormer 在做什么

Transformer 分割模型的通病是：token 序列越长 attention 越容易**塌缩**（attention collapse，
所有位置的输出趋同，等于丢掉了空间信息）。DTMFormer 用两个模块解决：

| 模块 | 全称 | 作用 |
| --- | --- | --- |
| **ATM** | Adaptive Token Merging | 按「相似度 + 重要性」把多个 token **融合**成一个，缩短序列、集中信息 |
| **LTR** | Learnable Token Reconstruction | 在解码端把被合并的 token **还原**回原始分辨率，补回空间细节 |

一句话：**先因为压缩而不得不 merge，再用 LTR 把丢掉的空间结构找回来。**

把它插到 SETR 上后：参数量 −18M，GFLOPs 53.2 → 9.8，ACDC 平均 Dice 92.53%（论文）。

### 1.2 为什么选它

同期还调研了同组的 APFormer（TMI 2022）。两篇面对**同一个问题**却走了相反的路：
APFormer 是自适应**剪枝**（把冗余 token 扔掉），DTMFormer 是**融合**（把相似 token 合起来）。
前者不可逆，后者可以还原——这个对比是本项目的切入点。

---

## 2. 环境

| 项目 | 版本 / 说明 |
| --- | --- |
| 平台 | Kaggle Notebook，Accelerator = **GPU T4**，Internet = **On** |
| Python | 3.13 |
| PyTorch | 2.x（CUDA） |
| 额外依赖 | `einops`、`thop`、`medpy`、`hausdorff`、`opencv-python`、`h5py`、`matplotlib` |
| 数据集 | Kaggle 公开数据集 `anhoangvo/acdc-dataset`（MIT，324MB，1912 个切片 h5） |
| 显存占用 | 推理阶段约 2GB，远低于 T4 上限 |

> `visualizer` **不要装**。PyPI 上有个同名包，但里面没有作者 import 的那个函数，
> 装了反而报错。见第四节坑 ③。

---

## 3. 完成到什么程度

| 环节 | 状态 | 证据 |
| --- | --- | --- |
| 环境搭建 | 完成 | Notebook 第 1–2 格，`import` 全部通过 |
| 官方权重恢复 | 完成 | 137MB 真实权重下载成功并 `load_state_dict` 无 Missing key |
| 模型结构正确性 | 完成 | thop 打印 **GFLOPs 9.784 / 参数量 33.527M**，与论文 9.8 GFLOPs 一致 |
| patient 级评估协议 | 完成 | `test.py --eval_mode patient`，输出 Dice + HD95 + mIoU + 灵敏度/特异度 |
| 数据转换与病人划分 | 完成 | 1841 张有效切片，100 个病人，train/test 严格按病人连续排列 |
| **复现论文指标** | **未达成** | 实测平均 Dice 0.493，论文 92.53%；已排除三个方向，见第五节 |

**这一行的意思**：模型搭得对、权重是对的那一份、评估协议是论文那一套——
三项都印证无误，所以差距不在「我有没有用对模型」这一层。

---

## 4. 踩坑记录

这一节是本仓库真正有价值的部分。六处坑全部来自**作者代码在新环境下的隐含假设**，
README 里一个字都没提。

| # | 现象 | 根因 | 解法 |
| --- | --- | --- | --- |
| ① | `torch.load` 拿到的权重无法加载 | 仓库里的 `Setr_DTMFormer.pth` 是 **Git LFS 指针**，只有 134 字节，直接 clone 拿到的是假文件 | 走 media 通道下载真实 137MB 文件：<br>`https://media.githubusercontent.com/media/iam-nacl/DTMFormer/main/checkpoints_ACDC/Setr_DTMFormer.pth` |
| ② | `UnpicklingError` / 加载失败 | PyTorch ≥ 2.6 起 `torch.load` 默认 `weights_only=True` | 补丁设为 False；注意**用默认参数绑定真函数**（见坑 ⑥） |
| ③ | `cannot import name 'get_local' from 'visualizer'` | 作者 import 了一个 PyPI 上**同名不同包**的东西；且 `get_local` 在全仓库从未被调用，是死代码 | 不装该包，改为注入占位模块到 `sys.modules` |
| ④ | `EinopsError: can't divide axis of length 154 in chunks of 8` | pipeline 里**没有任何 resize**，`--imgsize 256` 是障眼法——磁盘上的 PNG 必须已经是 256×256 | 生成 PNG 时统一 resize；图片用 `INTER_LINEAR`，**标签必须用 `INTER_NEAREST`** |
| ⑤ | 只评估也报 `FileNotFoundError` | 作者代码同时依赖 `trainofficial.txt` 和 `testofficial.txt`，且靠「文件名前 3 位 == 病人号」聚合成 patient | 文件名写成 `pid:03d_frame_slice.png`，列表严格按病人排序 |
| ⑥ | `RecursionError: maximum recursion depth exceeded` | Notebook 里 `_raw = torch.load` 存的是**全局变量**，重跑该单元格时它会被重新赋值成上一次的补丁本身，补丁自我调用 | 千萬别从 `torch.load` 取原函数。用 `torch.serialization.load`，并在定义时用默认参数绑死：<br>`def _patched_load(*a, _real=_TRUE_LOAD, **kw)` |

坑 ⑥ 是这一趟最隐蔽的一个：**同样的代码写成 `.py` 脚本没问题，放 Notebook 里跑两次就炸**。

---

## 5. 指标差距：我排除了什么

### 5.1 实测数字

论文 vs 我的实测（同为 patient 级协议，20 个测试病人，62 秒跑完）：

| 指标 | 论文 | 我的实测 |
| --- | --- | --- |
| 平均 Dice | 92.53% | **0.4930** |
| HD95 | — | 11.11 mm |
| RV（右心室） | — | 0.2289 |
| Myo（心肌） | — | 0.5501 |
| LV（左心室） | — | 0.7001 |
| GFLOPs | 9.8 | **9.7840** ✅ |
| 参数量 | −18M vs SETR | **33.527M** ✅ |

症状特征是**召回低、特异度极高**（灵敏度 0.17 / 0.46 / 0.65，特异度 0.999）——
典型的「该分割的地方没分割出来」。由此列出三个候选，逐个排除。

### 5.2 候选一：预处理与作者不一致 → **排除**

源码层面先确认：`ImageToImage2D` 读 PNG → `F.to_tensor` 直接进模型，
`z_score` 默认 False 且 `train.py` 与 `test.py` 构造参数一字不差，
**不存在训测不一致**。

然后做受控网格搜索（`diagnosis/preprocessing_grid.py`），4 种亮度 × 3 种几何：

| 亮度方案 | stretch | pad | crop |
| --- | --- | --- | --- |
| raw（原样 ×255） | 0.6095 | 0.5877 | 0.6074 |
| volmax（整帧统一） | 0.6107 | 0.5882 | 0.6078 |
| minmax（逐片拉伸） | 0.6105 | 0.5887 | 0.6076 |
| **perc（逐片分位）** | **0.6117** | 0.5907 | 0.6092 |

**12 个组合极差仅 0.024（0.588～0.612）**，而目标差距是 0.49 vs 0.92。

这个「打平」本身就是结论：如果错的是预处理，12 组里必然有一个明显冒尖。
全都挤在一起说明**模型对这些输入改动根本不敏感**，瓶颈不在输入侧。

> 附带一个被推翻的直觉：我一开始认为这批 h5 已经存成 0~1 float32，
> 那最自然的写法就是「原样 ×255」，它应该最好。实测 `raw = 0.6095`
> 反而略低于 `perc = 0.6117`。让数据说话，别让直觉说话。

### 5.3 候选二：图像与标签空间错位 → **排除**

先做配准自检：标签区域内平均亮度 0.5947，标签外 0.2012，**比值 2.96**。
远大于 1，说明数据本身的图像和标签是对齐的。

再做几何扰动搜索（`diagnosis/alignment_search.py`）：不动模型、不动图片，
只对真值做平移 / 翻转 / 旋转，看 Dice 会不会涨。

| 变换 | Dice | 相对基准 |
| --- | --- | --- |
| **原样（基准）** | **0.6186** | — |
| GT 左移 2px | 0.6185 | −0.0002 |
| GT 下移 2px | 0.5956 | −0.0231 |
| GT 左移 4px | 0.5723 | −0.0463 |
| GT 上移 2px | 0.5719 | −0.0467 |
| GT 右移 2px | 0.5651 | −0.0535 |
| GT 下移 4px | 0.5208 | −0.0979 |
| GT 上移 4px | 0.4988 | −0.1198 |
| GT 右移 4px | 0.4889 | −0.1297 |
| GT 上下翻转 | 0.4676 | −0.1510 |
| GT 左移 8px | 0.4457 | −0.1729 |
| GT 下移 8px | 0.3794 | −0.2392 |
| GT 上移 8px | 0.3668 | −0.2518 |
| GT 右移 8px | 0.3639 | −0.2547 |
| GT 左移 16px | 0.2881 | −0.3306 |
| GT 左右翻转 | 0.2833 | −0.3353 |
| GT 旋转 180° | 0.2684 | −0.3502 |
| GT 下移 16px | 0.2488 | −0.3699 |
| GT 右移 16px | 0.2405 | −0.3781 |
| GT 上移 16px | 0.2261 | −0.3926 |
| GT 左移 32px | 0.1158 | −0.5028 |
| GT 右移 32px | 0.0940 | −0.5246 |
| GT 下移 32px | 0.0938 | −0.5248 |
| GT 上移 32px | 0.0845 | −0.5341 |

**基准就是冠军，其余 23 项全部为负，且衰减极陡**（32px 已掉到 0.08）。
说明预测与真值在位置上已经是最准的，任何挪动都只会更差。

### 5.4 候选三：权重本身 → **当前结论**

三条排查线全部阴性，剩下的只能是：**官方放出的这份 checkpoint 达不到论文数字**
（早停 / 未收敛，或论文使用的是未公开的病人划分）。

论文没有公开 patient 划分方式，本项目取 100 个病人里**最后 20 个**作为测试集，
因此即便模型正常，数字也不会严格等于 92.53%。但 0.49 与 0.92 的差距量级
已经超出「划分不同」能解释的范围。

---

## 6. 待办

- [ ] 用 ACDC 官方 challenge 数据从头跑 `train.py`（默认 4000 epoch），对比自发权重
- [ ] 联系作者确认 patient 划分与训练轮数
- [ ] 补齐与 APFormer 的对比实验（APFormer 仓库无预训练权重，需自行训练）

---

## 7. 文件说明

| 文件 | 内容 |
| --- | --- |
| `DTMFormer_Kaggle_复现.ipynb` | 完整流程笔记本：环境 → 权重 → 数据 → 评估 → 可视化 |
| `diagnosis/preprocessing_grid.py` | 预处理受控网格搜索 + 配准自检 + 混淆矩阵 |
| `diagnosis/alignment_search.py` | 空间错位搜索 + 三联对比图 |
| `assets/` | 预测效果图、排查过程图 |
| `requirements.txt` | 依赖清单 |

---

## 8. 参考

- DTMFormer: *Delving into Token Transformation for Medical Image Segmentation* (AAAI 2024), DOI `10.1609/aaai.v38i6.28394`
- 官方代码：https://github.com/iam-nacl/DTMFormer
- APFormer（对比对象）：arXiv `2206.14413`，https://github.com/xianlin7/APFormer
- ACDC 数据集：https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
- Kaggle 镜像数据集：https://www.kaggle.com/datasets/anhoangvo/acdc-dataset

---

## 附：本项目的另一个产出

排障过程中沉淀出一套「复现指标差距」的排查流程，适用于任何
「官方代码跑通了但数字对不上」的场景：

> **先读源码钉死约定 → 再确认跑通（结构 / 权重 / 协议三项）→
> 抄症状缩小范围 → 受控网格搜索单个变量 → 网格拉不开就查空间错位 →
> 全部阴性则判定 checkpoint 质量。**

关键经验是：**网格搜索拉不开差距是有效结论，不是失败**。它把一个方向彻底关掉，
比「碰运气找到一个能用的组合」更有价值。
