# `load_static_gene_prior` 使用说明

`load_static_gene_prior` 用来把 scGPT 和 Geneformer 的 static gene embedding 按你的输入基因顺序对齐。它不会改动现有 NicheTrans 模型结构，只负责读取已经导出的 embedding、完成基因名称/物种映射，并返回可直接接入后续模型实验的 tensor。

## 前置条件

先确保已经运行过 embedding 构建脚本：

```powershell
D:\BINN\.conda\gene-prior-gpu\python.exe prior_AddOn\build_static_gene_embeddings.py --root D:\BINN\prior_AddOn\gene_embeddings --models scgpt geneformer
```

脚本会生成：

- `D:\BINN\prior_AddOn\gene_embeddings\processed\scgpt_static.pt`
- `D:\BINN\prior_AddOn\gene_embeddings\processed\geneformer_v2_316m_static.pt`
- `D:\BINN\prior_AddOn\gene_embeddings\processed\mapping_cache.tsv`
- `D:\BINN\prior_AddOn\gene_embeddings\manifest.json`

## 基本用法

```python
from prior_AddOn.gene_embedding_loader import load_static_gene_prior

priors = load_static_gene_prior(
    source_panel=dataset.source_panel,
    species="mouse",  # or "human"
    root=r"D:\BINN\prior_AddOn\gene_embeddings",
)

scgpt_prior = priors["scgpt"]
geneformer_prior = priors["geneformer"]
```

`source_panel` 的顺序会被严格保留。返回的 `embeddings[i]` 一定对应 `source_panel[i]`。

## 参数

```python
load_static_gene_prior(
    source_panel,
    species,
    models=("scgpt", "geneformer"),
    root=None,
    dataset_key=None,
    write_aligned=False,
    allow_network=True,
)
```

- `source_panel`：输入基因列表，通常直接传现有数据管理器里的 `dataset.source_panel`。
- `species`：`"human"` 或 `"mouse"`，也支持 `"homo_sapiens"`、`"mus_musculus"` 等别名。
- `models`：要加载的模型，默认同时加载 `"scgpt"` 和 `"geneformer"`。
- `root`：embedding 产物根目录；默认是 `prior_AddOn/gene_embeddings`。
- `dataset_key`：当 `write_aligned=True` 时用于输出文件名。
- `write_aligned`：是否把对齐后的结果保存到 `processed/aligned/{dataset_key}_{model}.pt`。
- `allow_network`：是否允许调用 Ensembl REST API 做 mouse-to-human ortholog 映射。设为 `False` 时只使用本地 `mapping_cache.tsv`。

## 返回结构

返回值是一个按模型名索引的字典：

```python
{
    "scgpt": {
        "embeddings": Tensor[n_genes, 512],
        "found_mask": BoolTensor[n_genes],
        "mapping_table": list[dict],
        "coverage": dict,
    },
    "geneformer": {
        "embeddings": Tensor[n_genes, 1152],
        "found_mask": BoolTensor[n_genes],
        "mapping_table": list[dict],
        "coverage": dict,
    },
}
```

字段含义：

- `embeddings`：按 `source_panel` 顺序排列的 embedding 矩阵。未映射到的基因对应全零向量。
- `found_mask`：布尔 mask，表示每个输入基因是否成功找到 embedding。
- `mapping_table`：逐基因映射记录，包含原始输入、标准化名称、映射 ID、token id、状态和原因。
- `coverage`：覆盖率统计，包括总特征数、成功映射数、覆盖比例和各状态计数。

## 常见状态

- `mapped`：成功找到 embedding。
- `missing_embedding`：基因能解析，但对应模型 vocab 中没有 embedding。
- `unmapped`：无法解析到目标基因 ID。
- `ambiguous`：存在多候选映射，默认不随机选择。
- `non_gene_feature`：输入像 ATAC peak 或特殊 token，不会伪造 gene embedding。
- `invalid_input`：空字符串或无效输入。

## 保存对齐结果

```python
priors = load_static_gene_prior(
    dataset.source_panel,
    species="human",
    root=r"D:\BINN\prior_AddOn\gene_embeddings",
    dataset_key="human_lymph_node",
    write_aligned=True,
)
```

这会额外写出：

- `processed/aligned/human_lymph_node_scgpt.pt`
- `processed/aligned/human_lymph_node_geneformer.pt`

保存文件中包含 `source_panel`、`embeddings`、`found_mask`、`mapping_table` 和 `coverage`。

## 映射规则

- Ensembl version suffix 会被去掉，例如 `ENSG00000141510.17` 会标准化为 `ENSG00000141510`。
- human symbol 会先 exact match，再尝试 uppercase fallback。
- Geneformer 优先使用 Ensembl ID；symbol 通过 Geneformer 字典转成 Ensembl ID。
- mouse symbol 或 `ENSMUSG...` 会通过 Ensembl homology 映射到 human one-to-one ortholog，再进入 scGPT/Geneformer vocab。
- ATAC peak 等非基因特征会标记为 `non_gene_feature`。

## 快速检查覆盖率

```python
for model_name, prior in priors.items():
    print(model_name, prior["coverage"])
```

也可以查看缺失项：

```python
for row in priors["geneformer"]["mapping_table"]:
    if row["status"] != "mapped":
        print(row["input_gene"], row["status"], row["reason"])
```

## 离线模式

如果不希望联网，使用：

```python
priors = load_static_gene_prior(
    dataset.source_panel,
    species="mouse",
    root=r"D:\BINN\prior_AddOn\gene_embeddings",
    allow_network=False,
)
```

此时 mouse-to-human 映射只会读取本地 `mapping_cache.tsv`。没有缓存的鼠基因会标记为 `unmapped`。

