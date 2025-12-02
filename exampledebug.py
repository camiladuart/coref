import torch
from transformers import AutoTokenizer
from coref_loader.training.model import CorefModel

# mini-doc artificial só pra depuração
sentences = [
    ["John", "went", "home"],
    ["He", "slept"]
]

gold_starts_all = torch.tensor([0, 3])   # "John", "He"
gold_ends_all   = torch.tensor([0, 3])

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

model = CorefModel({
    "encoder_name": "bert-base-cased",
    "max_span_width": 3,
    "max_segment_len": 2,
    "dropout": 0.2,
})

logits, labels, loss = model.forward(
    sentences=sentences,
    seg_start=0,
    seg_sents=sentences,
    tokenizer=tokenizer,
    gold_starts_all=gold_starts_all,
    gold_ends_all=gold_ends_all,
    max_span_width=3,
)

print("logits:", logits)
print("labels:", labels)
print("loss:", loss)

print("shape da matriz de pares:", model.last_pair_scores.shape)
print("primeiros 5x5 scores:")
print(model.last_pair_scores[:5, :5])