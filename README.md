# IPOP Eu 发射波长复现与泛化审计

本项目复现 Jang 等人在 Scientific Reports 发表的 IPOP 数据集 Eu 激活荧光粉发射波长基线，并比较随机行划分与配方、基质、文献分组划分。重点不是追逐单一最高 R²，而是判断模型面对未见材料体系时的真实泛化能力。

## 数据与许可

工作流下载 Figshare 固定版本的 IPOP v3 主表和 Eu 发射特征表，并在使用前检查文件大小、MD5 与 SHA-256。仅当 Figshare 返回 HTTP 403 时，下载器才回退至论文作者/发布方 KRICT 的官方 GitHub 镜像；镜像内容仍必须通过同一份固定 size、MD5 和 SHA-256 校验。`data/interim/source_provenance.json` 同时记录清单来源与实际 `retrieved_url`，不会把已验证的镜像缓存伪称为重新从 Figshare 下载。数据集采用 CC BY 4.0；下载后的原始数据、处理数据和实验输出均被 Git 忽略。

## 为什么不能声称逐数值复现

原论文没有公开完整的随机划分、随机种子、超参数搜索及所有平台细节。因此这里进行的是固定公开数据和固定评估协议下的方法级复现，而非逐小数位的宣称。报告会自动标示超出预先声明阈值的复现差异，且不会改变随机种子挑选结果。

## 环境安装

需要 Python 3.11 或更新版本。在项目根目录创建并启用虚拟环境，然后以开发模式安装：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux：

```bash
source .venv/bin/activate
```

```bash
python -m pip install -e ".[dev]"
```

## 一键运行

```bash
python -m ipop all
```

该命令按固定顺序下载、验证、准备数据、执行嵌套交叉验证并生成中文报告。XGBoost 使用固定的 `reg:squarederror`、`hist`、单线程和经批准网格。

## 分步运行

```bash
python -m ipop download
python -m ipop validate
python -m ipop prepare-emission
python -m ipop run
python -m ipop report
```

任一步失败都应先检查前一步的中间文件；`download` 可安全重跑，已存在的文件仍会重新校验哈希。

## 输出文件

`data/interim/source_provenance.json` 记录 DOI、许可、下载时间、来源链接和哈希；`validation.json` 记录主表不变量；`emission_prepared.csv` 和 `metadata_join_audit.csv` 记录匹配数据与文献歧义。实验目录 `outputs/emission-xgb/` 包含 splits、分组重叠审计、逐行预测、折级指标、汇总指标、最优参数和运行元数据；报告另外写出 `findings_zh.md` 与四张 PNG 图。

## 评估协议

所有协议均为 5 个外层折、3 个内层折，使用固定 seed 42。特征集合为 AF、AF+T、AF+ES、AF+T+ES；模型是中位数基线和 XGBoost。除 `random_row` 外，`group_formula`、`group_host`、`group_reference` 分别禁止同一配方、host 或文献在训练与测试折之间重叠。`overlap_audit.csv` 是这一约束的可检验记录。

## 结果解读

首先查看 `summary_metrics.csv` 中 `random_row / AF+T+ES / xgboost`：它适于与论文的随机行基线作方向性比较。随后比较三种分组协议，后者更接近对新材料体系的外推。报告还绘制协议性能与残差、随机行特征消融和随机行预测一致性图，并保留均值与折间标准差，避免只报告最佳折。

## 已知限制

Eu 发射样本来自主表匹配，少数行可存在多个来源 DOI；它们被审计标注，而不是任意指定一个文献。分组名称是数据集提供的元数据，不能保证完全等同于所有化学语义；结果也不能代替实验验证。跨环境的软件版本差异会带来数值浮动。

## 后续研究路线

可扩展到其他激活离子与性质目标，加入结构表征或不确定性量化，并优先采用按时间、文献或家族的前瞻性评估。任何新增模型应保持相同的外层分组折与公开的参数搜索记录，以便公平比较。

## 引用

请引用论文 DOI [10.1038/s41598-024-58351-w](https://doi.org/10.1038/s41598-024-58351-w) 与数据集 Figshare DOI [10.6084/m9.figshare.24771186.v1](https://doi.org/10.6084/m9.figshare.24771186.v1)。数据集采用 CC BY 4.0。
