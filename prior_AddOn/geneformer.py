import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM

model_name = "ctheodoris/Geneformer"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

with torch.no_grad():
    E = model.get_input_embeddings().weight.detach().cpu()

vocab = tokenizer.get_vocab()  # token -> id
id2token = {v: k for k, v in vocab.items()}

special_tokens = set(tokenizer.all_special_tokens)

rows = []
for token, idx in vocab.items():
    if token in special_tokens:
        continue
    rows.append((token, idx))

tokens = [t for t, i in rows]
indices = [i for t, i in rows]

gene_emb = E[indices]

df = pd.DataFrame(gene_emb.numpy(), index=tokens)
df.to_csv("geneformer_static_gene_embeddings.csv")

torch.save(
    {
        "genes_or_tokens": tokens,
        "embedding": gene_emb,
    },
    "geneformer_static_gene_embeddings.pt",
)