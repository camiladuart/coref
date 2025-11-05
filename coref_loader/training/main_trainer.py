#imprime instâncias do dataset
#terminal: python -m coref_loader.training.main_trainer --data_dir "C:\Users\PC\coref_data\ontonotes_onf" --split dev --limit 5 --preview 10
    
#dificuldades: 
#tentei usar o dataset original do OntoNotes, com o codigo ontonotes_to_jsonlines.py, mas tive problemas com formatos: ele vem em formato .onf e o ontonotes-db-tool oficial é em Python 2, então não consegui importar.
        #mudei de plano e passei a testar com arquivos .jsonlines criados manualmente.
#no setup inicial: tive muitos erros de ambiente (versões de python, tensorflow, etc) -> resolvi criando um ambiente limpo (python 3.7) e instalando manualmente o que precisava.
#imports e estrutura: o python não encontrava o pacote coref_loader -> resolvi rodando o trainer como módulo (python -m coref_loader.training.main_trainer).
#roda 100% em Python 3; consigo ler e imprimir instâncias do dataset no formato .jsonlines; o código está sem dependências antigas (tensorflow, pytorch, etc.)

import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer
from coref_loader.data import CorefDataset, flatten_sentences, build_candidates, extract_gold_spans
from coref_loader.training.model import CorefModel

def main():
    ap = argparse.ArgumentParser() #criando o leitor de argumentos. add os args que o programa vai aceitar:
    ap.add_argument("--data_dir", required=True, help="Pasta com *.english.jsonlines") #datadir
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"]) #split: train/dev/test
    ap.add_argument("--limit", type=int, default=10, help="Quantos docs usar (0 = todos)") #quantos docs quero ver
    args = ap.parse_args() #le o que foi digitado no terminal e guarda em args

    #config p escolher o modelo
    config = {
        "encoder_name": "bert-base-cased",
        "max_span_width": 30,
        "dropout": 0.2,
    }

    ds = CorefDataset(args.data_dir, args.split) #dados
    print(f"[OK] carregado: {len(ds)} documentos ({args.split})")

    tokenizer = AutoTokenizer.from_pretrained(config["encoder_name"]) #tokenizer
    model = CorefModel(config) #modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #se for train: otimizador
    if args.split == "train":
        optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

    used = 0
    model.train() if args.split == "train" else model.eval()

    for ex in ds:
        if args.limit and used >= args.limit:
            break
        used += 1

        # 1) Achatar sentenças e construir sentence_map
        tokens, sentence_map = flatten_sentences(ex["sentences"])

        # 2) Tokenizar (sem especiais), mantendo 1-para-1 com tokens originais
        enc = tokenizer(tokens, is_split_into_words=True, add_special_tokens=False, return_tensors="pt")
        input_ids = enc["input_ids"]          # [1, T]
        attention_mask = enc["attention_mask"]# [1, T]

        # 3) Candidatos (não cruzam sentença, largura <= max_span_width)
        span_starts, span_ends = build_candidates(sentence_map, config["max_span_width"])
        if span_starts.numel() == 0:
            print("Sem candidatos neste doc; pulando.")
            continue
        span_batch_idx = torch.zeros_like(span_starts)  # tudo no mesmo item do batch (B=1)

        # 4) Gold spans (0/1 por candidato)
        gold_starts, gold_ends = extract_gold_spans(ex)
        mention_labels = model.get_candidate_labels(span_starts, span_ends, gold_starts, gold_ends)

        # 5) Tensores para o device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        span_starts = span_starts.to(device)
        span_ends = span_ends.to(device)
        span_batch_idx = span_batch_idx.to(device)
        mention_labels = mention_labels.to(device)

        # 6) Forward + loss
        out = model.get_prediction_and_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_starts=span_starts,
            span_ends=span_ends,
            span_batch_idx=span_batch_idx,
            mention_labels=mention_labels
        )
        loss = out["loss"]

        if args.split == "train":
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        # Log simples
        with torch.no_grad():
            probs = torch.sigmoid(out["logits"])
            preds = (probs >= 0.5).long()
            acc = (preds == mention_labels).float().mean().item()
        print(f"doc {used}: T={input_ids.size(1)}, spans={span_starts.numel()}, loss={float(loss):.4f}, acc={acc:.3f}")

if __name__ == "__main__":
    main()

