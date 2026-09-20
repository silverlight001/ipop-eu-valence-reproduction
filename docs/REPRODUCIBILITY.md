# 复现指南

## 1. 环境

- Python 3.11+
- 支持 Windows、macOS 和 Linux
- 依赖在 `pyproject.toml` 中固定大版本范围
- XGBoost 训练固定单线程以减少跨运行差异

创建环境：

```bash
python -m venv .venv
```

激活后安装：

```bash
python -m pip install -e ".[dev]"
```

## 2. 验证仓库

```bash
python -m pytest -q
python -m ruff check src tests
```

当前公开版本预期：102 个测试通过，ruff 无错误。

## 3. 数据下载与校验

即使仓库已经包含固定原始 CSV，也可以重新下载并验证：

```bash
python -m ipop download
python -m ipop validate
```

下载器会验证文件大小、MD5 和 SHA-256。若 Figshare 返回 HTTP 403，只回退到 KRICT-DATA 官方镜像，并继续使用相同哈希验证。

生成的来源记录：

```text
data/interim/source_provenance.json
data/interim/validation.json
```

## 4. 准备 Eu 发射数据

```bash
python -m ipop prepare-emission
```

预期：

- 1665 行；
- Eu²⁺ 626 行；
- Eu³⁺ 1039 行；
- 6 行来源 DOI 歧义；
- 价态缺失/冲突 0 行。

输出：

```text
data/interim/emission_prepared.csv
data/interim/metadata_join_audit.csv
```

## 5. 原始复现

完整执行：

```bash
python -m ipop all
```

或在数据已准备时：

```bash
python -m ipop run
python -m ipop report
```

输出目录：`outputs/emission-xgb/`。

## 6. 价态比较

完整执行：

```bash
python -m ipop all-valence
```

或：

```bash
python -m ipop run-valence
python -m ipop report-valence
```

输出目录：`outputs/emission-valence-comparison/`。

训练采用固定单线程和嵌套 CV，可能需要较长时间。程序默认静默运行，完成前输出目录包含 `.incomplete`。

## 7. 完整性检查

成功运行后应满足：

- `.incomplete` 不存在；
- `fold_metrics.csv` 有 320 行；
- `summary_metrics.csv` 有 64 行；
- `predictions.csv` 有 106464 行；
- pooled 和 separate 各有 53232 条预测；
- 每个协议、特征、模型和 row_id 在每种模式中只出现一次；
- 两种模式的 row_id 与 fold 完全一致；
- `overlap_audit.csv` 中所有 enforced 行的 `overlap_count` 为 0。

## 8. 运行元数据

`run_metadata.json` 保存：

- seed；
- 完整实验配置；
- Eu²⁺/Eu³⁺样本数；
- Python、NumPy、pandas、scikit-learn 和 XGBoost 版本；
- 原始数据来源与哈希记录。

`best_params.json` 保存每个协议、特征集、模型、外层折和价态子模型的最优参数。

## 9. 数值差异

不同操作系统和依赖小版本可能造成轻微浮点差异。判断是否成功时优先检查：

1. 数据哈希和样本数；
2. 折分配是否一致；
3. pooled/separate 是否逐行配对；
4. 指标是否处于同一量级；
5. 是否存在强制分组键重叠。

不要通过更换随机种子或只报告最佳折来追求更高分。
