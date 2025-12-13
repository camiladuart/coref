#cd coref_main
#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#.\.venv\Scripts\Activate.ps1
#python -m coref_loader.training.main_trainer --data_dir "C:\Users\PC\coref_data\ontonotes_onf" --split dev --limit 5
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer
from coref_loader.data import CorefDataset, extract_gold_spans_with_clusters
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
        "max_segment_len": 3,  
        "top_span_ratio": 0.4,          #igual ao independent.py
        "max_top_antecedents": 50,      # c máximo -> valor p começar
        "use_genre": True,   #só usa gênero se estiver True
        "genres": ["bc", "bn", "nw", "pt", "wb", "tc", "mz", "cctv", "weibo", "forum", "news"], 
        "genre_emb_size": 20,  #=feature_size
    }

    # dataset:
    ds = CorefDataset(args.data_dir, args.split, config)
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

    # percorrer os docs do dataset
    for idx, ex in enumerate(ds):
        if args.limit and idx >= args.limit:
            break

        sentences = ex["sentences"]
        print("DOC_KEY:", ex.get("doc_key"))
        print("GENRE_EXTRAIDO:", ex.get("genre"))

        gold_starts_all, gold_ends_all, gold_cluster_ids_all = extract_gold_spans_with_clusters(ex)
        genre = ex.get("genre", None)

        chunk_no = 0  # contador de segmento dentro do doc - p numerar os segmentos dentor do doc

        # percorre o doc em segmentos de ate max_segment_len
        for seg_start, seg_sents in sentence_chunks(sentences, config["max_segment_len"]):
            chunk_no += 1

            #chamo a forward
            logits, mention_labels, loss = model.forward(
                sentences=sentences,
                seg_start=seg_start,
                seg_sents=seg_sents,
                tokenizer=tokenizer,
                gold_starts_all=gold_starts_all,
                gold_ends_all=gold_ends_all,
                gold_cluster_ids_all=gold_cluster_ids_all, 
                max_span_width=config["max_span_width"],
                genre=genre,
            )
            if logits.numel() == 0:
                continue

            # passo de treino:
            if args.split == "train":
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

            # p visualizar 
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                k_total = probs.numel()

                if k_total > 0:
                    preds = (probs >= 0.5).long()
                    acc = (preds == mention_labels).float().mean().item()
                    k = min(5, k_total)
                    topk_vals, _ = probs.topk(k)
                    top_str = ", ".join(f"{topk_vals[r].item():.3f}" for r in range(k))
                else:
                    acc = 0.0
                    top_str = "—"

            print(
                f"[doc {idx+1:02d} | seg {chunk_no:02d}] "
                f"loss={loss.detach().item():.4f} | acc={acc:.3f} | "
                f"spans={mention_labels.numel():4d} | "
                f"top5_scores=[{top_str}]"
            )           
if __name__ == "__main__":
    main()