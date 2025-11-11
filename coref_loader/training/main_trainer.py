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
    
# dividir o doc em segmentos (substitui a truncagem -> max_segment_len)
def sentence_chunks(sentences, max_segment_len):
    for i in range(0, len(sentences), max_segment_len):
        yield i, sentences[i : i + max_segment_len]

def main():
    ap = argparse.ArgumentParser()  # criando o leitor de argumentos. add os args que o programa vai aceitar:
    ap.add_argument("--data_dir", required=True, help="Pasta com *.english.jsonlines")  # datadir
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])  # split: train/dev/test
    ap.add_argument("--limit", type=int, default=10, help="Quantos docs usar (0 = todos)")  # quantos docs quero ver
    args = ap.parse_args()  # lê o que foi digitado no terminal e guarda em args

    # config p escolher o modelo
    config = {
        "encoder_name": "bert-base-cased",
        "max_span_width": 30,
        "max_segment_len": 3,  # adicionei
        "dropout": 0.2,
    }

    # dataset:
    ds = CorefDataset(args.data_dir, args.split)
    print(f"[OK] carregado: {len(ds)} documentos ({args.split})")

    # modelo + tokenizer + device:
    tokenizer = AutoTokenizer.from_pretrained(config["encoder_name"])  # carrega o tokenizer do BERT
    model = CorefModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # se for train: otimizador
    if args.split == "train":
        optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train() if args.split == "train" else model.eval()  # modelo em modo treino

    # p visualizar no terminal:
    tot_docs = 0
    tot_loss = 0.0
    tot_acc = 0.0  # rever no artigo métricas usadas!

    # percorrer os docs do dataset
    for idx, ex in enumerate(ds):
        if args.limit and idx >= args.limit:
            break

        sentences = ex["sentences"]
        gold_starts_all, gold_ends_all = extract_gold_spans(ex)

        chunk_no = 0  # contador de segmento dentro do doc - p numerar os segmentos dentor do doc

        # percorre o doc em segmentos de ate max_segment_len
        for seg_start, seg_sents in sentence_chunks(sentences, config["max_segment_len"]):
            chunk_no += 1

            # juntar sentenças e construir sentence_map
            tokens, sentence_map = flatten_sentences(seg_sents)
            if not tokens:
                continue

            # tokenizar: transf tokens em ids numéricos p/ encoder (com truncagem para 512) + criação da mask
            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                add_special_tokens=False,
                truncation=True,          # truncagem agora é por segmento (até 512 WPs), não no doc inteiro
                max_length=512,           # limite do BERT
                return_tensors="pt"
            )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]

            T_wp = input_ids.size(1)  # quantos subtokens tem este segmento

            #wordpiece
            try: #p garantir compatibilidade com versoes do tokenizer
                wp2tok = enc.word_ids(0)
            except Exception:
                wp2tok = enc.encodings[0].word_ids
            #construir primeiro/último WP de cada token 
            n_tokens = len(tokens)
            first_wp = [-1] * n_tokens #listas começam com -1
            last_wp = [-1] * n_tokens
            for wp_idx in range(T_wp):     #percorrer todos os wordpieces do segmento    
                tok_idx = wp2tok[wp_idx] #dizer qual token gerou esse wordpiece
                if tok_idx is None:
                    continue
                if first_wp[tok_idx] == -1:     # se for a primeira vez vendo o tok_idx, guarda em first
                    first_wp[tok_idx] = wp_idx
                last_wp[tok_idx] = wp_idx #sempre atualizo no final (utlimo wp que vi para esse token)
    

            # gerando candidatos (que não cruzam sentença, largura <= max_span_width)
            span_starts, span_ends = build_candidates(sentence_map, config["max_span_width"])
            if span_starts.numel() == 0:
                continue

            # filtrar gold spans do segmento
            offset_glob = sum(len(s) for s in sentences[:seg_start])#inicio do segmento (soma quantos tokens existem antes do segmento)
            if gold_starts_all.numel() > 0: #segue se ha gold spans
                keep = (gold_starts_all >= offset_glob) & (gold_ends_all < offset_glob + len(tokens)) #cria uma mascara booleana (keep) pra pegar só as mençoes que caem dentro do inertavalo desse segmento (gold_starts_all e gold_ends_all)
                gold_starts = gold_starts_all[keep] - offset_glob
                gold_ends = gold_ends_all[keep] - offset_glob
            else:
                gold_starts = gold_ends = torch.empty(0, dtype=torch.long)

            mention_labels = model.get_candidate_labels(span_starts, span_ends, gold_starts, gold_ends)

            # wordpiece: converter spans tokens-> wp usando first_wp/last_wp
            span_start_wp, span_end_wp = [], []
            for s, e in zip(span_starts.tolist(), span_ends.tolist()):
                fs, le = first_wp[s], last_wp[e]
                if 0 <= fs <= le < T_wp:
                    span_start_wp.append(fs)
                    span_end_wp.append(le)
            if not span_start_wp:
                continue

            # para garantir que todos os tensores estão no mesmo device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            span_start_wp = torch.tensor(span_start_wp, dtype=torch.long, device=device)
            span_end_wp = torch.tensor(span_end_wp, dtype=torch.long, device=device)
            span_batch_idx = torch.zeros_like(span_start_wp, device=device)
            mention_labels = mention_labels.to(device)

            # Forward + loss
            out = model.get_prediction_and_loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_starts=span_start_wp,   
                span_ends=span_end_wp,
                span_batch_idx=span_batch_idx,
                mention_labels=mention_labels
            )
            loss = out["loss"]

            # passo de treino:
            if args.split == "train":
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            # p visualizar 
            with torch.no_grad():
                probs = torch.sigmoid(out["logits"]) #converte os logits(números brutos) em probabilidades(0-1) usando sigmoid
                k_total = probs.numel() #tamanho (quantos spans existem nesse segmento)

                if k_total > 0: #se nao está vazio, faz as predições binarias
                    preds = (probs >= 0.5).long()
                    acc = (preds == mention_labels).float().mean().item() #compara a predição com o rotulo vdd
                    k = min(5, k_total)   #mostra top5 spans                           
                    topk_vals, topk_idx = probs.topk(k)
                    top_str = ", ".join(f"{topk_vals[r].item():.3f}" for r in range(k))
                else:
                    acc = 0.0
                    top_str = "—"  #caso nao haja spans no segmento                          

                print(
                    f"[doc {idx+1:02d} | seg {chunk_no:02d}] "
                    f"T_wp={input_ids.size(1):3d} | spans={span_start_wp.numel():4d} | "
                    f"loss={loss.detach().item():.4f} | acc={acc:.3f} | "
                    f"top5_scores=[{top_str}]"
                )
                
if __name__ == "__main__":
    main()