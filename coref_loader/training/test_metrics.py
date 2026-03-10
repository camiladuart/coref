import argparse
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer
from coref_loader.data import CorefDataset, extract_gold_spans_with_clusters
from coref_loader.training.model import CorefModel
from coref_loader.data import sentence_chunks
from scipy.optimize import linear_sum_assignment

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
def evaluate_test_metrics(model, test_ds, tokenizer, config, device, limit=0):
    model.eval()
    #acumuladores globais (eu estava fazendo media por doc -> metricas altas -> alteraçao:)
    total_muc_tp_p = 0
    total_muc_p = 0
    total_muc_tp_r = 0
    total_muc_r = 0
    total_b3_prec_num = 0
    total_b3_prec_den = 0
    total_b3_rec_num = 0
    total_b3_rec_den = 0
    total_ceaf_sim = 0
    total_ceaf_pred = 0
    total_ceaf_gold = 0

    with torch.no_grad():
        for doc_i, ex in enumerate(test_ds):
            if limit and doc_i >= limit:
                break

            sentences = ex["sentences"]
            gold_starts_all, gold_ends_all, gold_cluster_ids_all = extract_gold_spans_with_clusters(ex)
            genre = ex.get("genre", None)

            #add memória cross-segment (lembrar+relacionar span seg 1 com 2)
            memory_span_emb = []
            memory_mention_scores = []
            memory_segment_ids = []
            memory_beam_pred_ids = []
            memory_beam_sizes = []
            
            #gold clusters (start,end)
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

            sent_lens = [len(s) for s in sentences]
            sent_prefix = [0] #sent_prefix[k] = tokens antes da sentence k
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
                    speakers=ex.get("speakers", None),
                    return_debug=True,
                )

                dbg = getattr(model, "last_debug", None)
                if dbg is None:
                    continue

                curr_span_emb = dbg["span_emb"]
                curr_mention_scores = dbg["mention_scores"]
                curr_segment_ids = dbg["span_segment_ids"]
    
                #spans do beam (indices em tokens do segmento)
                starts_seg = dbg["span_starts_tok"].detach().cpu().tolist()
                ends_seg   = dbg["span_ends_tok"].detach().cpu().tolist()

                #offset de tokens antes deste segmento no doc
                seg_token_offset = sent_prefix[seg_start]
                
                probs = torch.sigmoid(logits).detach().cpu().tolist()
                beam_to_pred = {}   # índice do beam -> índice em pred_mentions
                for i, (s, e, p) in enumerate(zip(starts_seg, ends_seg, probs)):
                    if p >= 0.3:  # threshold de menção alterei de 0.5 - mt alto
                        beam_to_pred[i] = len(pred_mentions)
                        pred_mentions.append((seg_token_offset + s, seg_token_offset + e))

                if doc_i == 0 and seg_start == 0:
                    top = sorted([(p,i) for i,p in enumerate(probs)], reverse=True)[:10]
                    print("\nTop10 mention probs (prob, idx):", top)
                    print("Exemplo spans (idx, start,end,prob):")
                    for p,i in top[:5]:
                        print(i, starts_seg[i], ends_seg[i], p)
                        
                beam_pred_ids = [-1] * len(starts_seg)
                for i in range(len(starts_seg)):
                    if i in beam_to_pred:
                        beam_pred_ids[i] = beam_to_pred[i]

                #links: para cada span atual que passou o threshold, escolher melhor antecedente entre spans do seg atual e spans de segs anteriores
                curr_beam_size = len(starts_seg)

                #scores intra-segmento já calculados pelo modelo 
                intra_scores = dbg["pair_scores"].detach().cpu().numpy() 

                for i in range(curr_beam_size):
                    if i not in beam_to_pred:
                        continue

                    best_score = 0.0   #threshold: só linka se bater o dummy (score > 0)
                    best_pred_idx = -1

                    #1.antecedentes no mesmo segmento (intra)
                    if i > 0:
                        row_intra = intra_scores[i][:i]
                        j_intra = int(row_intra.argmax())
                        if row_intra[j_intra] > best_score:
                            if j_intra in beam_to_pred:
                                best_score = row_intra[j_intra]
                                best_pred_idx = beam_to_pred[j_intra]

                    #2.antecedentes em segmentos anteriores (cross-segment)
                    #calcula só as interações: span_i atual vs. todos os spans anteriores
                    if len(memory_span_emb) > 0:
                        prev_span_emb_cat = torch.cat(memory_span_emb, dim=0)      
                        prev_scores_cat   = torch.cat(memory_mention_scores, dim=0) 
                        prev_seg_ids_cat  = torch.cat(memory_segment_ids, dim=0)   

                        curr_emb_i = curr_span_emb[i].unsqueeze(0).to(prev_span_emb_cat.device)  
                        D = curr_emb_i.size(-1)
                        prev_total = prev_span_emb_cat.size(0)

                        #broadcasting só para 1 span vs. todos os anteriores 
                        emb_i_exp = curr_emb_i.expand(prev_total, D)
                        emb_j_exp = prev_span_emb_cat

                        #distância de segmento
                        seg_i_val = curr_segment_ids[i].item()
                        seg_j_vals = prev_seg_ids_cat
                        seg_dist = (torch.full_like(seg_j_vals, seg_i_val) - seg_j_vals)
                        seg_dist = seg_dist.clamp(0, model.max_training_sentences - 1)
                        seg_emb_cross = model.segment_distance_embeddings(seg_dist) 

                        cross_feats = [emb_i_exp, emb_j_exp, emb_i_exp * emb_j_exp, seg_emb_cross]

                        if model.use_speakers:
                            #speaker do span atual vs. todos os anteriores
                            spk_label = torch.full(
                                (prev_total,), 2, dtype=torch.long,
                                device=prev_span_emb_cat.device
                            )
                            spk_emb_cross = model.speaker_embeddings(spk_label)  # [prev_total, 20]
                            cross_feats.append(spk_emb_cross)

                        pair_input_cross = torch.cat(cross_feats, dim=-1)  

                        with torch.no_grad():
                            cross_scores = model.pair_scorer(pair_input_cross).squeeze(-1)  
                        cross_scores = cross_scores + curr_mention_scores[i] + prev_scores_cat
                        cross_scores = cross_scores.detach().cpu().numpy()

                        best_j_cross = int(cross_scores.argmax())
                        if cross_scores[best_j_cross] > best_score:
                            #mapear best_j_cross -> pred_idx
                            tmp = best_j_cross
                            seg_k = 0
                            while seg_k < len(memory_beam_sizes) and tmp >= memory_beam_sizes[seg_k]:
                                tmp -= memory_beam_sizes[seg_k]
                                seg_k += 1
                            if seg_k < len(memory_beam_pred_ids):
                                prev_pred_idx = memory_beam_pred_ids[seg_k][tmp]
                                if prev_pred_idx != -1:
                                    best_score = cross_scores[best_j_cross]
                                    best_pred_idx = prev_pred_idx

                    if best_pred_idx != -1:
                        pred_links.append((beam_to_pred[i], best_pred_idx))

                memory_span_emb.append(curr_span_emb)
                memory_mention_scores.append(curr_mention_scores)
                memory_segment_ids.append(curr_segment_ids)
                memory_beam_pred_ids.append(beam_pred_ids)
                memory_beam_sizes.append(curr_beam_size)               
            
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
            
            #Calculo das metricas pos alteracoes (sem ser media por doc -> numeros altissimos). MUC:
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

            mentions = set(gold_m2c.keys()) | set(pred_m2c.keys())

            for m in mentions:
                g = gold_m2c.get(m, {m})
                p_set = pred_m2c.get(m, {m})
                inter = len(g & p_set)

                total_b3_prec_num += inter
                total_b3_prec_den += len(p_set)

                total_b3_rec_num += inter
                total_b3_rec_den += len(g)

            #CEAFe com Hungarian matching
            if pred_clusters and gold_clusters:
                import numpy as np
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
    b3_prec = total_b3_prec_num / total_b3_prec_den if total_b3_prec_den>0 else 0.0
    b3_rec  = total_b3_rec_num  / total_b3_rec_den  if total_b3_rec_den>0 else 0.0
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
    ap.add_argument("--split", choices=["dev", "test"], default="test")  # <-- novo
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
        model, ds, tokenizer, config, device, limit=args.limit
    )

    print(f"\n{args.split.upper()} METRICS:")
    print(f"MUC    F1: {muc:.4f}")
    print(f"B³     F1: {b3:.4f}")
    print(f"CEAF-e F1: {ceaf:.4f}\n")


if __name__ == "__main__":
    main()