# 数据说明与来源追踪

## 1. 固定数据版本

本项目固定使用 IPOP dataset ver. 3.0 的 Figshare v1：

- DOI：<https://doi.org/10.6084/m9.figshare.24771186.v1>
- 作者官方镜像：<https://github.com/KRICT-DATA/IPOP-dataset-ver-3.0>
- 许可证：CC BY 4.0
- 数据论文：<https://doi.org/10.1038/s41598-024-58351-w>

`data/manifests/ipop_v3.json` 固定记录下载 URL、文件大小、MD5 和 SHA-256。工作流优先使用 Figshare；当 Figshare 返回 HTTP 403 时，只允许回退到 KRICT-DATA 官方 GitHub 镜像，且镜像文件必须通过相同哈希校验。

## 2. 原始文件

| 文件 | 作用 | 大小 | SHA-256 |
| --- | --- | ---: | --- |
| `Inorganic_Phosphor_Optical_Properties_DB_20230908_IPOP_ver3.csv` | IPOP 主表，含组成、dopant、价态、光学性质、条件与来源 | 987261 B | `9ebbee222e7b21faac0919761fbc8ee76c304c8ae3c8c89f0ab384a3d53b2924` |
| `phosphor_20230908_Eu_only_EmP_AF.csv` | Eu 发射波长任务的 52 维原子属性特征表 | 570738 B | `c573eee9a3893a8501be83f24807eac10f633d32d439c048c0cdc17fd8233afd` |

主表删除 437 条完全空白物理行后有 3952 条有效记录，包含 2238 个 host、553 篇来源文献和 16023 条性质观测。这些不变量由 `validate_master` 强制检查。

## 3. 关键字段

主表中的相关字段：

- `Inorganic phosphor`：含掺杂元素和浓度的完整名义组成；
- `Host`：作者整理的 host 组成；
- `1st dopant` / `2nd dopant`：第一、第二 dopant；
- `1st dopant valency` / `2nd dopant valency`：对应 dopant 价态；
- `Temp. (K)`：测量温度；
- `Excitation source (nm)`：激发波长；
- `Emission max. (nm)`：最大 PL 发射波长；
- `Reference`：来源 DOI；
- `MP-ID` / `ICSD-ID`：结构数据库映射元数据。

Eu-only 特征表包含 52 个组成统计特征、温度、激发波长、发射峰和 `Formula`，但不包含 Eu 价态。这正是原始 ML 特征表中 Eu²⁺/Eu³⁺被混合的原因。

## 4. 价态回连

Eu-only 特征表通过以下四个观测身份字段与主表匹配：

1. `Formula` ↔ `Inorganic phosphor`
2. `Temp. (K)`
3. `Excitation source (nm)`
4. `Emission max. (nm)`

`Emission max. (nm)`只用于确认原始观测行身份，不作为推断价态的规则。模型训练时目标值不会被加入输入特征。

匹配程序执行以下硬检查：

- 每条记录必须匹配唯一的 host；
- Eu 必须出现在第一或第二 dopant；
- Eu 价态必须是 2 或 3；
- 同一观测的重复主表记录必须对价态达成一致；
- 来源 DOI 若不一致，保留为歧义并在 `group_reference` 评估中排除。

最终结果：

| 类别 | 行数 |
| --- | ---: |
| 全部 Eu 发射样本 | 1665 |
| Eu²⁺ | 626 |
| Eu³⁺ | 1039 |
| 价态缺失或冲突 | 0 |
| 来源 DOI 有歧义 | 6 |

## 5. 派生文件

`data/interim/`：

- `source_provenance.json`：DOI、许可、实际下载来源、时间和哈希；
- `validation.json`：主表公开不变量；
- `emission_prepared.csv`：带 `row_id`、Host、Reference 和 Eu valence 的建模表；
- `metadata_join_audit.csv`：逐行记录匹配数量并检查 host、DOI 与价态一致性。

`outputs/emission-valence-comparison/`：

- `splits.csv`：每种协议的外层折分配；
- `overlap_audit.csv`：Formula、Host、Reference 的训练/测试重叠；
- `predictions.csv`：所有逐样本 OOF 预测；
- `fold_metrics.csv`：总体折级指标；
- `valence_fold_metrics.csv`：价态分层折级指标；
- `summary_metrics.csv` / `valence_summary_metrics.csv`：折级聚合；
- `best_params.json`：每个外层折和模型的最佳参数；
- `run_metadata.json`：配置、版本和样本计数；
- `final_results*.csv`：主结果对比表。

## 6. 已知数据限制

- 名义化学式不包含完整缺陷、占位和电荷补偿信息；
- 相同名义组成可能对应不同结构或价态；
- 文献数据倾向于报道成功材料，存在发表偏倚；
- 不同实验室的仪器、拟合和制样流程可能形成域偏移；
- 当前数据只保存最大峰值，而非完整发射光谱；
- `Host` 字符串不同不保证材料家族真正独立。

## 7. 许可

原始 IPOP 文件及其再分发遵循 CC BY 4.0。使用数据时必须引用数据 DOI 和原始数据论文。完整声明见仓库根目录的 `THIRD_PARTY_DATA.md`。
