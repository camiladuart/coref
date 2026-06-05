import argparse
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from coref_loader.data import CorefDataset, extract_gold_spans_with_clusters, sentence_chunks
from coref_loader.training.model import CorefModel
from coref_loader.training.test_metrics import normalize_speakers_for_doc

def predict_clusters_for_doc(model, ex, tokenizer, config, device, mention_thresh=0.0):
    sentences = ex["sentences"]
    raw_speakers = normalize_speakers_for_doc(ex.get("speakers", None), sentences)
    gold_starts_all, gold_ends_all, gold_cluster_ids_all = extract_gold_spans_with_clusters(ex)
    genre = ex.get("genre", None)

    sent_lens = [len(s) for s in sentences]
    sent_prefix = [0]
    for L in sent_lens:
        sent_prefix.append(sent_prefix[-1] + L)

    memory_span_emb = []
    memory_mention_scores = []
    memory_segment_ids = []
    memory_beam_pred_ids = []
    memory_beam_sizes = []
    memory_speaker_ids = []

    pred_mentions = []
    pred_links = []

    with torch.no_grad():
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
                speakers=raw_speakers,
                return_debug=True,
            )

            dbg = getattr(model, "last_debug", None)
            if dbg is None:
                continue

            curr_span_emb = dbg["span_emb"]
            curr_mention_scores = dbg["mention_scores"]
            curr_segment_ids = dbg["span_segment_ids"]
            starts_seg = dbg["span_starts_tok"].detach().cpu().tolist()
            ends_seg   = dbg["span_ends_tok"].detach().cpu().tolist()

            curr_speaker_ids = None
            if model.use_speakers and raw_speakers is not None:
                doc_speaker_flat = [spk for sent_spk in raw_speakers for spk in sent_spk]
                doc_unique_spk = list(dict.fromkeys(doc_speaker_flat))
                doc_spk_to_id = {s: idx for idx, s in enumerate(doc_unique_spk)}
                seg_sents_local = seg_sents
                seg_speaker_flat = [
                    spk
                    for sent_idx, sent_spk in enumerate(raw_speakers)
                    if seg_start <= sent_idx < seg_start + len(seg_sents_local)
                    for spk in sent_spk
                ]
                raw_ids = []
                for s in starts_seg:
                    if 0 <= s < len(seg_speaker_flat):
                        raw_ids.append(doc_spk_to_id.get(seg_speaker_flat[s], -1))
                    else:
                        raw_ids.append(-1)
                curr_speaker_ids = torch.tensor(raw_ids, dtype=torch.long, device=curr_span_emb.device)

            seg_token_offset = sent_prefix[seg_start]
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            beam_to_pred = {}
            for i, (s, e, p) in enumerate(zip(starts_seg, ends_seg, probs)):
                if p >= mention_thresh:
                    beam_to_pred[i] = len(pred_mentions)
                    pred_mentions.append((seg_token_offset + s, seg_token_offset + e))

            intra_scores = dbg["pair_scores"].detach().cpu().numpy()
            curr_beam_size = len(starts_seg)
            beam_pred_ids = [-1] * curr_beam_size
            for i in range(curr_beam_size):
                if i in beam_to_pred:
                    beam_pred_ids[i] = beam_to_pred[i]

            for i in range(curr_beam_size):
                if i not in beam_to_pred:
                    continue
                best_score = 0.0
                best_pred_idx = -1

                if i > 0:
                    row_intra = intra_scores[i][:i]
                    for j_intra in np.argsort(-row_intra):
                        if row_intra[j_intra] <= best_score:
                            break
                        if j_intra in beam_to_pred:
                            best_score = row_intra[j_intra]
                            best_pred_idx = beam_to_pred[j_intra]
                            break

                if len(memory_span_emb) > 0:
                    prev_emb = torch.cat(memory_span_emb, dim=0)
                    prev_scores = torch.cat(memory_mention_scores, dim=0)
                    prev_segs = torch.cat(memory_segment_ids, dim=0)
                    curr_i = curr_span_emb[i].unsqueeze(0).expand(prev_emb.size(0), -1)
                    seg_dist = (torch.full_like(prev_segs, curr_segment_ids[i].item()) - prev_segs).clamp(0, model.max_training_sentences - 1)
                    seg_emb = model.segment_distance_embeddings(seg_dist)
                    feats = [curr_i, prev_emb, curr_i * prev_emb, seg_emb]
                    if model.use_speakers:
                        prev_spk = torch.cat(memory_speaker_ids, dim=0)
                        spk_i = curr_speaker_ids[i].item() if curr_speaker_ids is not None else -1
                        if spk_i < 0:
                            spk_label = torch.full((prev_emb.size(0),), 2, dtype=torch.long, device=prev_emb.device)
                        else:
                            same = (prev_spk == spk_i).long()
                            unknown = (prev_spk < 0).long()
                            spk_label = torch.where(unknown == 1, torch.full_like(same, 2), 1 - same)
                        feats.append(model.speaker_embeddings(spk_label.to(prev_emb.device)))
                    cross_scores = model.pair_scorer(torch.cat(feats, dim=-1)).squeeze(-1)
                    cross_scores = (cross_scores + curr_mention_scores[i] + prev_scores).detach().cpu().numpy()
                    for best_j in np.argsort(-cross_scores):
                        if cross_scores[best_j] <= best_score:
                            break
                        tmp = best_j
                        for seg_k, bsz in enumerate(memory_beam_sizes):
                            if tmp < bsz:
                                prev_pred_idx = memory_beam_pred_ids[seg_k][tmp]
                                if prev_pred_idx != -1:
                                    best_score = cross_scores[best_j]
                                    best_pred_idx = prev_pred_idx
                                break
                            tmp -= bsz

                if best_pred_idx != -1:
                    pred_links.append((beam_to_pred[i], best_pred_idx))

            memory_span_emb.append(curr_span_emb)
            memory_mention_scores.append(curr_mention_scores)
            memory_segment_ids.append(curr_segment_ids)
            memory_beam_pred_ids.append(beam_pred_ids)
            memory_beam_sizes.append(curr_beam_size)
            memory_speaker_ids.append(curr_speaker_ids if curr_speaker_ids is not None
                                      else torch.full((curr_beam_size,), -1, dtype=torch.long, device=curr_span_emb.device))

    # agrupar em clusters via union-find
    n = len(pred_mentions)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pred_links:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    clusters = [[pred_mentions[i] for i in grp] for grp in groups.values() if len(grp) >= 2]
    return clusters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--save_dir", required=True)
    ap.add_argument("--n_docs", type=int, default=15)
    ap.add_argument("--max_clusters", type=int, default=50)
    ap.add_argument("--output_file", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_epoch = int((Path(args.save_dir) / "best_epoch.txt").read_text().strip())
    ckpt = torch.load(Path(args.save_dir) / f"checkpoint_epoch_{best_epoch}.pt", map_location=device)
    config = ckpt["config"]

    model = CorefModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["encoder_name"])
    ds = CorefDataset(args.data_dir, "test", config)

    all_tokens_per_doc = []
    for ex in ds:
        all_tokens_per_doc.append([tok for sent in ex["sentences"] for tok in sent])

    lines = []
    total = 0

    for doc_idx in range(min(args.n_docs, len(ds))):
        if total >= args.max_clusters:
            break
        ex = ds[doc_idx]
        all_tokens = all_tokens_per_doc[doc_idx]
        clusters = predict_clusters_for_doc(model, ex, tokenizer, config, device)

        lines.append(f"\n{'='*60}")
        lines.append(f"DOC {doc_idx}: {ex.get('doc_key', '?')}")
        lines.append(f"{'='*60}")

        doc_count = 0
        for cid, cluster in enumerate(clusters):
            if total >= args.max_clusters:
                break
            mentions = []
            for (s, e) in cluster:
                if 0 <= s <= e < len(all_tokens):
                    mentions.append(f'"{" ".join(all_tokens[s:e+1])}"')
                else:
                    mentions.append(f"(idx {s}-{e})")
            lines.append(f"  Cluster {cid+1}: {' | '.join(mentions)}")
            total += 1
            doc_count += 1
        lines.append(f"  [{doc_count} clusters neste doc]")

    lines.append(f"\nTOTAL: {total} clusters analisados")
    Path(args.output_file).write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {total} clusters em {args.output_file}")

if __name__ == "__main__":
    main()
