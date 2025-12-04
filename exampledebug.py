import torch
from transformers import AutoTokenizer
from coref_loader.training.model import CorefModel

sentences = [
    ["John", "went", "home"],
    ["He", "slept"],
]

# spans ouro em TOKENS (do documento inteiro)
# menção 1 = "John" (token 0)
# menção 2 = "He"   (token 3)
gold_starts_all = torch.tensor([0, 3])
gold_ends_all   = torch.tensor([0, 3])

# preciso dizer em que cluster cada menção está
# se as duas se referem à mesma pessoa ("John" = "He"),
# então elas estão no MESMO cluster -> id = 1
gold_cluster_ids_all = torch.tensor([1, 1], dtype=torch.long)
#clusters diferentes, seria por ex.: [1, 2]

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

config = {
    "encoder_name": "bert-base-cased",
    "max_span_width": 3,
    "max_segment_len": 2,
    "dropout": 0.2,
}

model = CorefModel(config)

# chamando a forward igual ao main_trainer
logits, mention_labels, loss = model.forward(
    sentences=sentences,
    seg_start=0,
    seg_sents=sentences, 
    tokenizer=tokenizer,
    gold_starts_all=gold_starts_all,
    gold_ends_all=gold_ends_all,
    gold_cluster_ids_all=gold_cluster_ids_all,
    max_span_width=config["max_span_width"],
)

print("logits:", logits)
print("labels:", mention_labels)
print("loss:", loss)
print("shape da matriz de pares:", model.last_pair_scores.shape)
print("primeiros 5x5 scores:")
print(model.last_pair_scores[:5, :5])