import argparse
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer
from coref_loader.data import CorefDataset, extract_gold_spans_with_clusters
from coref_loader.training.model import CorefModel
from coref_loader.data import sentence_chunks
from scipy.optimize import linear_sum_assignment

#funcao auxiliar p normalizar speakers (formato):
def normalize_speakers_for_doc(speakers, sentences):
    if speakers is None:
        return None

    if sentences is None or not isinstance(sentences, list):
        return None

    if (
        isinstance(speakers, list)
        and len(speakers) == len(sentences)
        and all(isinstance(x, list) for x in speakers)
    ):
        normalized = []
        for sent_tokens, sent_spk in zip(sentences, speakers):
            if sent_spk is None:
                normalized.append(["UNK"] * len(sent_tokens))
                continue
            if len(sent_spk) == len(sent_tokens):
                normalized.append([str(s) if s is not None else "UNK" for s in sent_spk])
            elif len(sent_spk) == 1:
                spk = str(sent_spk[0]) if sent_spk[0] is not None else "UNK"
                normalized.append([spk] * len(sent_tokens))
            else:
                return None
        return normalized

    if isinstance(speakers, list) and all(not isinstance(x, list) for x in speakers):
        flat_tokens = [tok for sent in sentences for tok in sent]
        if len(speakers) == len(flat_tokens):
            normalized = []
            pos = 0
            for sent_tokens in sentences:
                n = len(sent_tokens)
                chunk = speakers[pos:pos+n]
                normalized.append([str(s) if s is not None else "UNK" for s in chunk])
                pos += n
            return normalized

    if isinstance(speakers, list) and len(speakers) == len(sentences):
        if all(not isinstance(x, list) for x in speakers):
            normalized = []
            for sent_tokens, spk in zip(sentences, speakers):
                speaker_name = str(spk) if spk is not None else "UNK"
                normalized.append([speaker_name] * len(sent_tokens))
            return normalized

    return None

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

    gold_mentions = set(gold_m2c.keys())
    if not gold_mentions:
        return 0.0
    prec_sum = 0.0
    rec_sum = 0.0
    for m in gold_mentions:
        g = gold_m2c[m]
        p = pred_m2c.get(m, {m})
        inter = len(g & p)
        prec_sum += inter / len(p)
        rec_sum  += inter / len(g)

    prec = prec_sum / len(gold_mentions)
    rec  = rec_sum  / len(gold_mentions)
    f1 = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
    return f1

#CEAF-e = similaridade baseada em clusters
def ceaf_e_f1(pred_clusters, gold_clusters):
    if not pred_clusters and not gold_clusters:
        return 0.0

    #matriz de similaridade:
    sim = np.zeros((len(pred_clusters), len(gold_clusters)), dtype=np.float64)
    for i, pc in enumerate(pred_clusters):
        pc_set = set(pc)
        for j, gc in enumerate(gold_clusters):
            sim[i, j] = len(pc_set & set(gc))

    #hungarian matching: linear_sum_assignment minimiza -> negamos para maximizar
    row_ind, col_ind = linear_sum_assignment(-sim)
    total = sim[row_ind, col_ind].sum()

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

#uso na evaluate p MUC:
def muc_counts(clusters, other_m2c):
    tp = 0
    p = 0
    for cl in clusters:
        if len(cl) <= 1:
            continue
        p += len(cl) - 1
        parts = set()
        for m in cl:
            parts.add(other_m2c.get(m, ("singleton", m)))
        tp += (len(cl) - len(parts))
    return tp, p

#funçao evaluate:
def evaluate_test_metrics(model, test_ds, tokenizer, config, device, limit=0, mention_thresh=0.0):
    model.eval()
    #acumuladores globais 
    total_muc_tp_p = 0
    total_muc_p = 0
    total_muc_tp_r = 0
    total_muc_r = 0
    total_b3_prec_sum = 0.0  
    total_b3_rec_sum  = 0.0  
    total_b3_count    = 0     
    total_ceaf_sim = 0
    total_ceaf_pred = 0
    total_ceaf_gold = 0

    with torch.no_grad():
        for doc_i, ex in enumerate(test_ds):
            if limit and doc_i >= limit:
                break

            sentences = ex["sentences"]
            raw_speakers = normalize_speakers_for_doc(ex.get("speakers", None), sentences)
            gold_starts_all, gold_ends_all, gold_cluster_ids_all = extract_gold_spans_with_clusters(ex)
            genre = ex.get("genre", None)
            
            #gold clusters 
            gold_by_cid = {}
            for s, e, cid in zip(gold_starts_all.tolist(),
                                gold_ends_all.tolist(),
                                gold_cluster_ids_all.tolist()):
                if cid <= 0:
                    continue
                gold_by_cid.setdefault(cid, []).append((s, e))
            gold_clusters = list(gold_by_cid.values())
            
            #achatar gold para set de menções
            gold_mentions = [m for cl in gold_clusters for m in cl]
            gold_set = set(gold_mentions)

            if doc_i == 0:
                print("\n===== DOC 0 checando =====", flush=True)
                print("Exemplo gold mentions:", gold_mentions[:20], flush=True)

            logits, mention_labels, loss = model.forward_document(
                sentences=sentences,
                tokenizer=tokenizer,
                gold_starts_all=gold_starts_all,
                gold_ends_all=gold_ends_all,
                gold_cluster_ids_all=gold_cluster_ids_all,
                max_span_width=config["max_span_width"],
                genre=genre,
                speakers=raw_speakers,
                return_debug=True,
            )

            if logits.numel() == 0:
                continue

            dbg = model.last_debug
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            starts_all = dbg["span_starts_tok"].tolist()
            ends_all   = dbg["span_ends_tok"].tolist()

            pred_mentions = []
            pred_links    = []
            beam_to_pred  = {}

            for i, (s, e, p) in enumerate(zip(starts_all, ends_all, probs)):
                if p >= mention_thresh:
                    beam_to_pred[i] = len(pred_mentions)
                    pred_mentions.append((s, e))

            # antecedentes: ler directamente de pair_scores (já calculado globalmente)
            pair_scores_full = dbg["pair_scores"]  # tensor [N_beam, N_beam]
            N_beam = pair_scores_full.size(0)

            for i in range(N_beam):
                if i not in beam_to_pred:
                    continue
                row = pair_scores_full[i]
                sorted_j = torch.argsort(row, descending=True)
                best_score = 0.0
                best_pred_idx = -1
                for j in sorted_j.tolist():
                    score = row[j].item()
                    if score <= best_score:
                        break
                    if j in beam_to_pred and j < i:
                        best_score = score
                        best_pred_idx = beam_to_pred[j]
                        break
                if best_pred_idx != -1:
                    pred_links.append((beam_to_pred[i], best_pred_idx))
            
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
                
                gold_set_end_minus1 = set((s, e-1) for (s, e) in gold_set)
                overlap2 = len(pred_set & gold_set_end_minus1)
                print("Overlap se gold_end-1:", overlap2, flush=True)
                pred_set_end_minus1 = set((s, e-1) for (s, e) in pred_set)
                overlap3 = len(pred_set_end_minus1 & gold_set)
                print("Overlap se pred_end-1:", overlap3, flush=True)
                print("================================\n", flush=True) 

            # se não tiver menções, vira tudo zero
            if len(pred_mentions) == 0:
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
            
            #accumulate muc counts
            tp_p, p = muc_counts(pred_clusters, _flatten_clusters(gold_clusters))
            tp_r, r = muc_counts(gold_clusters, _flatten_clusters(pred_clusters))
            total_muc_tp_p += tp_p
            total_muc_p += p
            total_muc_tp_r += tp_r
            total_muc_r += r
            #b3:
            gold_m2c = {}
            for cl in gold_clusters:
                for m in cl:
                    gold_m2c[m] = set(cl)

            pred_m2c = {}
            for cl in pred_clusters:
                for m in cl:
                    pred_m2c[m] = set(cl)

            # B³ standard: só menções gold
            for m in set(gold_m2c.keys()):
                g = gold_m2c[m]
                p_set = pred_m2c.get(m, {m})   # singleton se não foi predita
                inter = len(g & p_set)
                total_b3_prec_sum += inter / len(p_set)
                total_b3_rec_sum  += inter / len(g)
                total_b3_count    += 1

            #CEAFe com Hungarian matching
            if pred_clusters and gold_clusters:
                sim_mat = np.zeros((len(pred_clusters), len(gold_clusters)), dtype=np.float64)
                for i, pc in enumerate(pred_clusters):
                    pc_set = set(pc)
                    for j, gc in enumerate(gold_clusters):
                        sim_mat[i, j] = len(pc_set & set(gc))
                row_ind, col_ind = linear_sum_assignment(-sim_mat)
                total_ceaf_sim += sim_mat[row_ind, col_ind].sum()

            total_ceaf_pred += sum(len(c) for c in pred_clusters)
            total_ceaf_gold += sum(len(c) for c in gold_clusters)
            
    #calculos finais: MUC:
    muc_prec = total_muc_tp_p / total_muc_p if total_muc_p > 0 else 0.0
    muc_rec  = total_muc_tp_r / total_muc_r if total_muc_r > 0 else 0.0
    muc_f1 = (2*muc_prec*muc_rec/(muc_prec+muc_rec)) if (muc_prec+muc_rec)>0 else 0.0
    #b3:
    b3_prec = total_b3_prec_sum / total_b3_count if total_b3_count > 0 else 0.0
    b3_rec  = total_b3_rec_sum  / total_b3_count if total_b3_count > 0 else 0.0
    b3_f1 = (2*b3_prec*b3_rec/(b3_prec+b3_rec)) if (b3_prec+b3_rec)>0 else 0.0
    #ceafe:
    ceaf_prec = total_ceaf_sim / total_ceaf_pred if total_ceaf_pred>0 else 0.0
    ceaf_rec  = total_ceaf_sim / total_ceaf_gold if total_ceaf_gold>0 else 0.0
    ceaf_f1 = (2*ceaf_prec*ceaf_rec/(ceaf_prec+ceaf_rec)) if (ceaf_prec+ceaf_rec)>0 else 0.0

    return muc_f1, b3_f1, ceaf_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", choices=["dev", "test"], default="test")  
    ap.add_argument("--mention_thresh", type=float, default=0.0, help="Threshold de menção (0.0 = todos os spans do beam)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA?", torch.cuda.is_available(), "| device:", device)

    # Se não passar resume_from, usa best_epoch.txt dentro do save_dir
    if not args.resume_from:
        best_path = Path(args.save_dir) / "best_epoch.txt"
        if not best_path.exists():
            raise FileNotFoundError(
                f"Não encontrado {best_path}. Rode primeiro o DEV para gerar best_epoch.txt."
            )
        best_epoch = int(best_path.read_text().strip())
        args.resume_from = str(Path(args.save_dir) / f"checkpoint_epoch_{best_epoch}.pt")
        print(f"[AUTO] resume_from = {args.resume_from}")

    # Carrega checkpoint primeiro para pegar o config real do treino
    checkpoint = torch.load(args.resume_from, map_location="cpu")
    if "config" not in checkpoint:
        raise KeyError("Checkpoint não tem 'config'. Re-treine com código que salva config no checkpoint.")
    config = checkpoint["config"]

    # Dataset do split escolhido
    ds = CorefDataset(args.data_dir, args.split, config)
    print(f"[OK] {args.split} docs: {len(ds)}")

    # Detectar gêneros com base no dataset (se ativado no config)
    if config.get("use_genre", False):
        all_genres = sorted({ex.get('genre') for ex in ds.samples if ex.get('genre')})
        config["genres"] = all_genres
        print(f"[OK] genres: {len(all_genres)}")

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(config["encoder_name"])
    model = CorefModel(config).to(device)

    # Carregar pesos
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[OK] Loaded checkpoint: {args.resume_from}")

    muc, b3, ceaf = evaluate_test_metrics(
        model, ds, tokenizer, config, device,
        limit=args.limit,
        mention_thresh=args.mention_thresh
    )

    print(f"\n{args.split.upper()} METRICS:")
    print(f"MUC    F1: {muc:.4f}")
    print(f"B³     F1: {b3:.4f}")
    print(f"CEAF-e F1: {ceaf:.4f}\n")


if __name__ == "__main__":
    main()
