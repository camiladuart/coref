#cd coref_main
#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#.\.venv\Scripts\Activate.ps1
#python -m coref_loader.training.main_trainer --data_dir "C:\Users\PC\coref_data\ontonotes_onf" --split dev --limit 5
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

    tokenizer = AutoTokenizer.from_pretrained(config["encoder_name"]) #carrega o tokenizer do BERT
    model = CorefModel(config) #modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #se for train: otimizador
    if args.split == "train":
        optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

    used = 0
    model.train() if args.split == "train" else model.eval() #modelo em modo treino
    
    #p visualizar no terminal:
    tot_docs = 0
    tot_loss = 0.0
    tot_acc  = 0.0

    #percorrer os docs do dataset
    for ex in ds:
        if args.limit and used >= args.limit:
            break
        used += 1

        #juntar sentenças e construir sentence_map
        tokens, sentence_map = flatten_sentences(ex["sentences"])

        #tokenizar: transf tokens em ids numericos p encoder (com truncagem para 512)+ criação da mask
        enc = tokenizer(
            tokens,
            is_split_into_words=True,
            add_special_tokens=False,
            truncation=True,         # corta para o max_length
            max_length=512,          # limite do BERT
            return_tensors="pt"
        )
        input_ids = enc["input_ids"]          
        attention_mask = enc["attention_mask"]
        
        # Mapeamento de wordpiece -> token original
        if hasattr(enc, "word_ids"):
            wp2tok = enc.word_ids(0)
        else:
            wp2tok = enc.encodings[0].word_ids
        #quantos tokens originais couberam dentro dos 512 wordpieces?
        # (maior índice de token aparecendo em wp2tok)
        tok_max_kept = max([i for i in wp2tok if i is not None]) if any(x is not None for x in wp2tok) else -1
        # trunca também tokens e sentence_map para manter alinhamento com os WPs
        if tok_max_kept + 1 < len(tokens):
            tokens = tokens[:tok_max_kept + 1]
            sentence_map = sentence_map[:tok_max_kept + 1]

        #gerando candidatos (q não cruzam sentença, largura <= max_span_width)
        span_starts, span_ends = build_candidates(sentence_map, config["max_span_width"])
        if span_starts.numel() == 0:
            print("Sem candidatos neste doc; pulando.")
            continue
        span_batch_idx = torch.zeros_like(span_starts)
        
        #converter spans para índices de wordpiece
        #para cada token i, é preciso o primeiro e o último WP daquele token -> construção de duas tabelas: first_wp[i], last_wp[i]
        T_wp = input_ids.size(1)
        first_wp = [-1] * (tok_max_kept + 1)
        last_wp  = [-1] * (tok_max_kept + 1)
        for wp_idx, tok_idx in enumerate(wp2tok[:T_wp]):
            if tok_idx is None:
                continue
            if first_wp[tok_idx] == -1:
                first_wp[tok_idx] = wp_idx
            last_wp[tok_idx] = wp_idx

        #mapeando cada span token->wp
        span_start_wp = []
        span_end_wp = []
        valid_mask = []
        for s, e in zip(span_starts.tolist(), span_ends.tolist()):
            fs = first_wp[s]
            le = last_wp[e]
            #manter apenas spans que ficaram totalmente dentro dos 512 WPs
            if fs != -1 and le != -1 and fs <= le:
                span_start_wp.append(fs)
                span_end_wp.append(le)
                valid_mask.append(1)
            else:
                valid_mask.append(0)

        #filtrar spans inválidos (que cortaram na truncagem)
        valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
        if valid_mask.sum() == 0:
            print("Todos os spans cortados pela truncagem; pulando doc.")
            continue
        span_starts = span_starts[valid_mask]
        span_ends = span_ends[valid_mask]
        span_batch_idx = span_batch_idx[valid_mask]
        span_start_wp = torch.tensor(span_start_wp, dtype=torch.long)
        span_end_wp   = torch.tensor(span_end_wp,   dtype=torch.long)
        

        #extrair gold spans e criar os rotulos 0/1 por candidato
        gold_starts, gold_ends = extract_gold_spans(ex)
        # filtra gold para a parte mantida (≤ tok_max_kept)
        keep_gold = (gold_ends <= tok_max_kept)
        gold_starts = gold_starts[keep_gold]
        gold_ends   = gold_ends[keep_gold]
        mention_labels = model.get_candidate_labels(span_starts, span_ends, gold_starts, gold_ends)

        #para garantir que todos os tensores estao no mesmo device
        # to device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        span_starts_wp = span_start_wp.to(device)
        span_ends_wp   = span_end_wp.to(device)
        span_batch_idx = span_batch_idx.to(device)
        mention_labels = mention_labels.to(device)

        #Forward + loss
        out = model.get_prediction_and_loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            span_starts=span_starts_wp,   #agora vai em WPs
            span_ends=span_ends_wp,       
            span_batch_idx=span_batch_idx,
            mention_labels=mention_labels
        )
        loss = out["loss"]

        #passo de treino:
        if args.split == "train":
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            
        #log (p visualizar)
        with torch.no_grad():
            probs = torch.sigmoid(out["logits"])     
            preds = (probs >= 0.5).long()             
            acc = (preds == mention_labels).float().mean().item() if mention_labels.numel() > 0 else 0.0
        #imprime linha
        print(f"doc {used}: T={input_ids.size(1)}, spans={span_starts.numel()}, loss={loss.detach().item():.4f}, acc={acc:.3f}")
        #acumula para médias
        tot_docs += 1
        tot_loss += float(loss)
        tot_acc  += acc
        
        #resumo final no terminal
        if tot_docs > 0:
            print("-----------------------------------------------")
            print(f"média ({tot_docs} docs): loss={tot_loss/tot_docs:.4f}, acc={tot_acc/tot_docs:.3f}")

if __name__ == "__main__":
    main()
