"""
Gera ficheiro CoNLL-U com predições do modelo para o scorer oficial.
Uso:
  python3 predict_conllu.py \
    --data_dir .../coref_data/coref_pt \
    --gold_conllu .../coref_data/coref_pt_conllu/test.conllu \
    --output_conllu .../predictions_corefpt_test.conllu \
    --save_dir .../checkpoints_romance_pt
"""
import argparse, torch, re
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer
from coref_loader.data import CorefDataset, extract_gold_spans_with_clusters
from coref_loader.training.model import CorefModel
from coref_loader.training.test_metrics import (
    evaluate_test_metrics, normalize_speakers_for_doc, union_find
)

def predict_one_doc(model, ex, tokenizer, config, device, mention_thresh=0.0):
    sentences = ex["sentences"]
    speakers = normalize_speakers_for_doc(ex.get("speakers", None), sentences)
    gold_starts, gold_ends, gold_cluster_ids = extract_gold_spans_with_clusters(ex)
    genre = ex.get("genre", None)

    with torch.no_grad():
        logits, _, _ = model.forward_document(
            sentences=sentences,
            tokenizer=tokenizer,
            gold_starts_all=gold_starts,
            gold_ends_all=gold_ends,
            gold_cluster_ids_all=gold_cluster_ids,
            max_span_width=config["max_span_width"],
            genre=genre,
            speakers=speakers,
        )

    if logits.numel() == 0:
        return []

    dbg = model.last_debug
    probs = torch.sigmoid(logits).cpu().tolist()
    starts = dbg["span_starts_tok"].tolist()
    ends   = dbg["span_ends_tok"].tolist()

    beam_to_pred = {}
    pred_mentions = []
    for i, (s, e, p) in enumerate(zip(starts, ends, probs)):
        if p >= mention_thresh:
            beam_to_pred[i] = len(pred_mentions)
            pred_mentions.append((s, e))

    pair_scores = dbg["pair_scores"]
    pred_links = []
    for i in range(pair_scores.size(0)):
        if i not in beam_to_pred:
            continue
        row = pair_scores[i]
        for j in torch.argsort(row, descending=True).tolist():
            score = row[j].item()
            if score <= 0.0:
                break
            if j in beam_to_pred and j < i:
                pred_links.append((beam_to_pred[i], beam_to_pred[j]))
                break

    clusters_idx = union_find(len(pred_mentions), pred_links)
    return [[pred_mentions[i] for i in grp] for grp in clusters_idx if len(grp) >= 2]

def clusters_to_token_map(clusters):
    opens   = defaultdict(list)
    closes  = defaultdict(list)
    singles = defaultdict(list)
    for cid, cluster in enumerate(clusters, 1):
        for (s, e) in cluster:
            if s == e:
                singles[s].append(cid)
            else:
                opens[s].append(cid)
                closes[e].append(cid)
    token_map = {}
    for tok in set(list(opens) + list(closes) + list(singles)):
        tags = []
        for cid in singles[tok]: tags.append(f"(e{cid})")
        for cid in opens[tok]:   tags.append(f"(e{cid}")
        for cid in closes[tok]:  tags.append(f"e{cid})")
        token_map[tok] = "|".join(tags)
    return token_map

def write_predictions(gold_conllu, output_path, all_clusters):
    lines = Path(gold_conllu).read_text(encoding="utf-8").splitlines()
    doc_idx = -1
    tok_in_doc = 0
    token_map = {}
    out = []

    for line in lines:
        if line.startswith("# newdoc"):
            doc_idx += 1
            tok_in_doc = 0
            clusters = all_clusters[doc_idx] if doc_idx < len(all_clusters) else []
            token_map = clusters_to_token_map(clusters)
            out.append(line)
            continue

        if line.startswith("#") or line.strip() == "":
            out.append(line)
            continue

        fields = line.split("\t")
        # skip multi-word tokens and empty nodes
        if len(fields) < 10 or "." in fields[0] or "-" in fields[0]:
            out.append(line)
            continue

        misc_parts = [p for p in fields[9].split("|") if not p.startswith("Entity=")]
        tag = token_map.get(tok_in_doc, "")
        if tag:
            misc_parts.append(f"Entity={tag}")
        fields[9] = "|".join(misc_parts) if misc_parts else "_"
        out.append("\t".join(fields))
        tok_in_doc += 1

    Path(output_path).write_text("\n".join(out), encoding="utf-8")
    print(f"[OK] Predições escritas: {output_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",      required=True)
    ap.add_argument("--gold_conllu",   required=True)
    ap.add_argument("--output_conllu", required=True)
    ap.add_argument("--save_dir",      required=True)
    ap.add_argument("--resume_from",   default=None)
    ap.add_argument("--split",         default="test", choices=["dev","test"])
    ap.add_argument("--mention_thresh",type=float, default=0.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA:", torch.cuda.is_available())

    if not args.resume_from:
        best_epoch = int((Path(args.save_dir) / "best_epoch.txt").read_text().strip())
        args.resume_from = str(Path(args.save_dir) / f"checkpoint_epoch_{best_epoch}.pt")
    print(f"[AUTO] resume_from = {args.resume_from}")

    ckpt   = torch.load(args.resume_from, map_location=device)
    config = ckpt["config"]

    tokenizer = AutoTokenizer.from_pretrained(config["encoder_name"])
    model     = CorefModel(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[OK] Modelo carregado: {args.resume_from}")

    ds = CorefDataset(args.data_dir, args.split, config)
    print(f"[OK] {len(ds)} docs no split '{args.split}'")

    all_clusters = []
    for i, ex in enumerate(ds):
        clusters = predict_one_doc(model, ex, tokenizer, config, device, args.mention_thresh)
        all_clusters.append(clusters)
        if i % 5 == 0:
            print(f"  [{i+1}/{len(ds)}] clusters: {len(clusters)}")

    write_predictions(args.gold_conllu, args.output_conllu, all_clusters)

if __name__ == "__main__":
    main()
