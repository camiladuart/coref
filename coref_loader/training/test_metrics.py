import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer
from coref_loader.data import CorefDataset, extract_gold_spans_with_clusters
from coref_loader.training.model import CorefModel
from coref_loader.training.main_trainer import sentence_chunks

#criar mapa - a qual cluster cada span pertence: 
def _flatten_clusters(clusters):
    m2c = {} #mention to cluster
    for ci, cl in enumerate(clusters): #ci: n do cluster; cl: lista de mençoes do cluster
        for m in cl:
            m2c[m] = ci #mençao m pertence ao cluster ci
    return m2c #dicionario final

#metrica MUC (mede se o modelo colocou as mencoes certas juntas):
def muc_f1(pred_clusters, gold_clusters):
    def muc_counts(clusters, other_m2c):
        tp = 0 #quantos links o modelo acertou
        p = 0 #n total de links
        for cl in clusters:
            if len(cl) <= 1:
                continue
            p += len(cl) - 1 #max de links possiveis no cluster
            #ve se as mençoes foram separadas em outros clusters:
            parts = set()
            for m in cl:
                parts.add(other_m2c.get(m, ("singleton", m)))
            tp += (len(cl) - len(parts))
        return tp, p

    gold_m2c = _flatten_clusters(gold_clusters)
    pred_m2c = _flatten_clusters(pred_clusters)
    
    #precision e recall:
    tp_p, p = muc_counts(pred_clusters, gold_m2c)
    tp_r, r = muc_counts(gold_clusters, pred_m2c)
    
    prec = tp_p / p if p > 0 else 0.0
    rec  = tp_r / r if r > 0 else 0.0
    f1 = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
    return f1

#B3 (precision e recall por mençao)
def b3_f1(pred_clusters, gold_clusters):
    gold_m2c = {} #cria dicionario p cluster gold da mençao:
    for cl in gold_clusters:
        for m in cl:
            gold_m2c[m] = set(cl)
            
    pred_m2c = {} #cria dicionario com o que o modelo previu:
    for cl in pred_clusters:
        for m in cl:
            pred_m2c[m] = set(cl)

    mentions = set(gold_m2c.keys()) | set(pred_m2c.keys()) #junta as mençoes do gold e previstas
    if not mentions:
        return 0.0

    prec_sum = 0.0
    rec_sum = 0.0
    for m in mentions:
        g = gold_m2c.get(m, {m})
        p = pred_m2c.get(m, {m})
        inter = len(g & p) #mençoes que coincidem g e p 
        prec_sum += inter / len(p)
        rec_sum  += inter / len(g)

    prec = prec_sum / len(mentions)
    rec  = rec_sum  / len(mentions)
    f1 = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
    return f1

#CEAF-e = similaridade baseada em clusters
def ceaf_e_f1(pred_clusters, gold_clusters):
    if not pred_clusters and not gold_clusters:
        return 0.0

    #similarity matrix (compara cluster pred com cluster gold)
    sim = []
    for pc in pred_clusters:
        pc_set = set(pc)
        row = []
        for gc in gold_clusters:
            row.append(len(pc_set & set(gc)))
        sim.append(row)

    used_g = set() #clusters gold ja usados
    total = 0 #soma da similaridade
    for i in range(len(pred_clusters)): #para cada cluster previsto, escolher o melhor cluster gold:
        best_j = -1
        best = -1
        for j in range(len(gold_clusters)):
            if j in used_g:
                continue
            if sim[i][j] > best:
                best = sim[i][j]
                best_j = j
        if best_j >= 0 and best > 0:
            used_g.add(best_j)
            total += best

    pred_denom = sum(len(c) for c in pred_clusters)
    gold_denom = sum(len(c) for c in gold_clusters)
    prec = total / pred_denom if pred_denom > 0 else 0.0
    rec  = total / gold_denom if gold_denom > 0 else 0.0
    f1 = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
    return f1


#função union pra agrupar mençoes nos clusters:
def union_find(n, links):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for a, b in links:
        union(a, b)
    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    return list(groups.values())

#funçao evaluate:
def evaluate_test_metrics(model, test_ds, tokenizer, config, device, limit=0):
    model.eval()

    doc_muc = []
    doc_b3 = []
    doc_ceaf = []

    with torch.no_grad():
        for doc_i, ex in enumerate(test_ds):
            if limit and doc_i >= limit:
                break

            sentences = ex["sentences"]
            gold_starts_all, gold_ends_all, gold_cluster_ids_all = extract_gold_spans_with_clusters(ex)
            genre = ex.get("genre", None)

            # gold clusters (start,end)
            gold_by_cid = {}
            for s, e, cid in zip(gold_starts_all.tolist(),
                                gold_ends_all.tolist(),
                                gold_cluster_ids_all.tolist()):
                if cid <= 0:
                    continue
                gold_by_cid.setdefault(cid, []).append((s, e))
            gold_clusters = list(gold_by_cid.values())
            
            # achatar gold para set de menções
            gold_mentions = [m for cl in gold_clusters for m in cl]
            gold_set = set(gold_mentions)

            if doc_i == 0:
                print("\n===== DOC 0 checando =====", flush=True)
                print("Exemplo gold mentions:", gold_mentions[:20], flush=True)

            sent_lens = [len(s) for s in sentences]
            sent_prefix = [0] # sent_prefix[k] = tokens antes da sentence k
            for L in sent_lens:
                sent_prefix.append(sent_prefix[-1] + L)
        

            pred_mentions = []   #spans pred em lista
            pred_links = []  #links entre índices dessa lista (i->j)

            for seg_start, seg_sents in sentence_chunks(sentences, config["max_segment_len"]):
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
                    return_debug=True,
                )

                dbg = getattr(model, "last_debug", None)
                if dbg is None:
                    continue

                # spans do beam (indices em tokens do segmento)
                starts_seg = dbg["span_starts_tok"].detach().cpu().tolist()
                ends_seg   = dbg["span_ends_tok"].detach().cpu().tolist()

                # offset de tokens antes deste segmento no doc
                seg_token_offset = sent_prefix[seg_start]
                
                probs = torch.sigmoid(logits).detach().cpu().tolist()
                beam_to_pred = {}   # índice do beam -> índice em pred_mentions
                for i, (s, e, p) in enumerate(zip(starts_seg, ends_seg, probs)):
                    if p >= 0.5:  # threshold de menção
                        beam_to_pred[i] = len(pred_mentions)
                        pred_mentions.append((seg_token_offset + s, seg_token_offset + e))

                if doc_i == 0 and seg_start == 0:
                    top = sorted([(p,i) for i,p in enumerate(probs)], reverse=True)[:10]
                    print("\nTop10 mention probs (prob, idx):", top)
                    print("Exemplo spans (idx, start,end,prob):")
                    for p,i in top[:5]:
                        print(i, starts_seg[i], ends_seg[i], p)

                # links:
                pair = dbg["pair_scores"].detach().cpu().numpy()
                for i in range(1, len(starts_seg)):
                    if i not in beam_to_pred:
                        continue
                    row = pair[i][:i]
                    j = int(row.argmax()) if len(row) > 0 else -1
                    if j in beam_to_pred and row[j] > 0.5:
                        pred_links.append((beam_to_pred[i], beam_to_pred[j]))
                        
            #debug:
            if doc_i == 0:
                print("\n===== DEBUG DOC 0 =====")
                print("Gold clusters:", gold_clusters[:5], flush=True)
                print("Num gold clusters:", len(gold_clusters), flush=True)
                print("Num pred mentions:", len(pred_mentions), flush=True)
                print("Num pred links:", len(pred_links), flush=True)

            #TESTES debug: 
            pred_set = set(pred_mentions)
            overlap = len(pred_set & gold_set)
            if doc_i == 0:
                print("Overlap pred∩gold (same start, end):", overlap, flush=True)
                #TESTES (estava dando tudo 0): se eram erros de +1 -1 nos spans start, end:
                gold_set_end_minus1 = set((s, e-1) for (s, e) in gold_set)
                overlap2 = len(pred_set & gold_set_end_minus1)
                print("Overlap se gold_end-1:", overlap2, flush=True)
                pred_set_end_minus1 = set((s, e-1) for (s, e) in pred_set)
                overlap3 = len(pred_set_end_minus1 & gold_set)
                print("Overlap se pred_end-1:", overlap3, flush=True)
                print("================================\n", flush=True) #deu 0 ent ok 

            # se não tiver menções, vira tudo zero
            if len(pred_mentions) == 0:
                doc_muc.append(0.0); doc_b3.append(0.0); doc_ceaf.append(0.0)
                continue

            # clusters preditos por union-find (em índices)
            pred_clusters_idx = union_find(len(pred_mentions), pred_links)

            # converte clusters de idx -> spans (start,end) para comparar com gold
            pred_clusters = []
            for cl in pred_clusters_idx:
                pred_clusters.append([pred_mentions[i] for i in cl])
            #criando lista de tokens do doc: 
            doc_tokens = []
            for sent in sentences:
                doc_tokens.extend(sent)

            #quero ver o primeiro doc
            if doc_i == 0:
                print("\n========== DEBUG DOC 0 ==========")

                print("\nGOLD CLUSTERS:")
                for i, cl in enumerate(gold_clusters):
                    print(f"Gold {i}: {cl}")
                    for (s, e) in cl:
                        span_text = " ".join(doc_tokens[s:e+1])
                        print(f"  ({s},{e}) -> \"{span_text}\"")

                print("\nPREDICTED CLUSTERS:")
                for i, cl in enumerate(pred_clusters):
                    print(f"Pred {i}: {cl}")
                    for (s, e) in cl:
                        span_text = " ".join(doc_tokens[s:e+1])
                        print(f"  ({s},{e}) -> \"{span_text}\"")

                print("\n=================================\n")
            
            #métricas por doc
            doc_muc.append(muc_f1(pred_clusters, gold_clusters))
            doc_b3.append(b3_f1(pred_clusters, gold_clusters))
            doc_ceaf.append(ceaf_e_f1(pred_clusters, gold_clusters))

    # média no dataset
    MUC = sum(doc_muc)/len(doc_muc) if doc_muc else 0.0
    B3  = sum(doc_b3)/len(doc_b3) if doc_b3 else 0.0
    CEAF= sum(doc_ceaf)/len(doc_ceaf) if doc_ceaf else 0.0
    return MUC, B3, CEAF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    config = {
        "encoder_name": "bert-base-cased",
        "max_span_width": 30,
        "max_segment_len": 3,
        "top_span_ratio": 0.4,
        "max_top_antecedents": 50,
        "use_genre": True,
        "genres": [],
        "genre_emb_size": 20,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA?", torch.cuda.is_available(), "| device:", device)

    #TEST
    test_ds = CorefDataset(args.data_dir, "test", config)
    print(f"[OK] test docs: {len(test_ds)}")

    #detectar gêneros
    if config.get("use_genre", False):
        all_genres = sorted({ex.get("genre") for ex in test_ds.samples if ex.get("genre")})
        config["genres"] = all_genres
        print(f"[OK] genres: {len(all_genres)}")

    #tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(config["encoder_name"])
    model = CorefModel(config).to(device)

    #TEST: evaluate best checkpoint
    if not args.resume_from:
        best_path = Path(args.save_dir) / "best_epoch.txt"
        if not best_path.exists():
            raise FileNotFoundError(
                f"Não encontrado {best_path}. Rode primeiro o DEV para gerar best_epoch.txt."
            )

        best_epoch = int(best_path.read_text().strip())
        args.resume_from = str(Path(args.save_dir) / f"checkpoint_epoch_{best_epoch}.pt")
        print(f"[TEST] Auto resume_from = {args.resume_from}")

    checkpoint = torch.load(args.resume_from, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[TEST] Loaded checkpoint: {args.resume_from}")

    muc, b3, ceaf = evaluate_test_metrics(
        model, test_ds, tokenizer, config, device, limit=args.limit
    )

    print("\nTEST METRICS:")
    print(f"MUC   F1:  {muc:.4f}")
    print(f"B³    F1:  {b3:.4f}")
    print(f"CEAF-e F1: {ceaf:.4f}")
    print("\n")

if __name__ == "__main__":
    main()