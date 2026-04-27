"""
用 BiomedCLIP 将 patient_text_prompts.json 中的文本编码为 512 维 embedding，
保存为 text_embeddings.pt 供 SAM-Med2D 训练时查表使用。

从 modelscope 下载的本地模型直接加载，不需要联网。

Usage:
    pip install transformers
    python encode_text_embeddings.py

Output:
    /home/fym/Nas/fym/datasets/graduation/sam-med2d/text_embeddings.pt
"""

import json
import os
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizerFast

INPUT_JSON = "/home/fym/graduation/process_text/patient_text_prompts.json"
OUTPUT_PT = "/home/fym/Nas/fym/datasets/graduation/sam-med2d/text_embeddings.pt"
LOCAL_MODEL_DIR = "/home/fym/.cache/modelscope/hub/models/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class BiomedCLIPTextEncoder(nn.Module):
    """BiomedCLIP 文本编码器：BertModel + MLP 投影 (768 -> 640 -> 512)"""

    def __init__(self, bert: BertModel):
        super().__init__()
        self.bert = bert
        self.proj = nn.Sequential(
            nn.Linear(768, 640, bias=False),
            nn.GELU(),
            nn.Linear(640, 512, bias=False),
        )

    def forward(self, input_ids, attention_mask=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = out.last_hidden_state[:, 0]
        return self.proj(cls_token)


def load_text_encoder(model_dir):
    bert_cfg = BertConfig(
        vocab_size=30522, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072,
        max_position_embeddings=512,
    )
    bert = BertModel(bert_cfg)
    encoder = BiomedCLIPTextEncoder(bert)

    ckpt = torch.load(
        os.path.join(model_dir, "open_clip_pytorch_model.bin"),
        map_location="cpu",
    )

    state = {}
    for k, v in ckpt.items():
        if k.startswith("text.transformer."):
            state[k.replace("text.transformer.", "bert.")] = v
        elif k == "text.proj.0.weight":
            state["proj.0.weight"] = v
        elif k == "text.proj.2.weight":
            state["proj.2.weight"] = v

    encoder.load_state_dict(state, strict=False)
    return encoder


def main():
    with open(INPUT_JSON) as f:
        texts = json.load(f)
    print(f"Loaded {len(texts)} patient texts")

    tokenizer = BertTokenizerFast.from_pretrained(LOCAL_MODEL_DIR)
    encoder = load_text_encoder(LOCAL_MODEL_DIR)
    encoder.eval()
    print("Model and tokenizer loaded from local files (no network)")

    embeddings = {}
    with torch.no_grad():
        for i, (pid, text) in enumerate(texts.items()):
            tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
            feat = encoder(tokens["input_ids"], tokens["attention_mask"])
            feat = feat / feat.norm(dim=-1, keepdim=True)
            embeddings[pid] = feat.squeeze(0).cpu()
            if (i + 1) % 20 == 0:
                print(f"  Encoded {i+1}/{len(texts)}")

    torch.save(embeddings, OUTPUT_PT)
    sample_key = list(embeddings.keys())[0]
    print(f"Saved {len(embeddings)} embeddings to {OUTPUT_PT}")
    print(f"Sample shape: {embeddings[sample_key].shape}")


if __name__ == "__main__":
    main()