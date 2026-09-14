import argparse
import os
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer

from coref_loader.data import CorefDataset, extract_gold_spans_with_clusters
from coref_loader.training.model import CorefModel
from coref_loader.training.test_metrics import normalize_speakers_for_doc, union_find


def resolve_local_encoder_path(encoder_name):
    """
    Resolve um modelo Hugging Face para um snapshot local no cache.
    Isso evita tentativa de acesso à internet no nó do Slurm.
    """
    p = Path(encoder_name)
    if p.exists():
        return str(p)

    hf_home = Path(os.environ.get(
        "HF_HOME",
        "/projects/F202600026AIVLABDEUCALION/up202000683/hf_cache"
    ))

    hub_cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", hf_home / "hub"))

    model_cache_name = "models--" + encoder_name.replace("/", "--")
    snapshots_dir = hub_cache / model_cache_name / "snapshots"

    if snapshots_dir.exists():
        snapshots = [x for x in snapshots_dir.iterdir() if x.is_dir()]
        if snapshots:
            # usa o snapshot mais recente
            snapshots = sorted(snapshots, key=lambda x: x.stat().st_mtime, reverse=True)
            local_path = snapshots[0]
            print(f"[LOCAL ENCODER] {encoder_name} -> {local_path}")
            return str(local_path)

    raise FileNotFoundError(
        f"Could not find local Hugging Face snapshot for {encoder_name}. "
        f"Checked: {snapshots_dir}"
    )


def predict_one_doc(model, ex, tokenizer, config, device, mention_thresh=0.0):
    """
    Roda o modelo em um documento e devolve clusters previstos
    no formato: [[(start, end), (start, end), ...], ...]
    usando offsets flat de token no documento.
    """
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
            return_debug=True,
        )

    if logits.numel() == 0:
        return []

    dbg = model.last_debug
    if dbg is None:
        raise RuntimeError(
            "model.last_debug is None. forward_document must be called with return_debug=True."
        )

    probs = torch.sigmoid(logits).detach().cpu().tolist()
    starts = dbg["span_starts_tok"].tolist()
    ends = dbg["span_ends_tok"].tolist()
    pair_scores = dbg["pair_scores"]

    beam_to_pred = {}
    pred_mentions = []

    for i, (s, e, p) in enumerate(zip(starts, ends, probs)):
        if p >= mention_thresh:
            beam_to_pred[i] = len(pred_mentions)
            pred_mentions.append((int(s), int(e)))

    if len(pred_mentions) == 0:
        return []

    pred_links = []

    for i in range(pair_scores.size(0)):
        if i not in beam_to_pred:
            continue

        row = pair_scores[i]

        for j in torch.argsort(row, descending=True).tolist():
            score = float(row[j].item())

            #antecedent links only when pair score is positive
            if score <= 0.0:
                break

            if j in beam_to_pred and j < i:
                pred_links.append((beam_to_pred[i], beam_to_pred[j]))
                break

    clusters_idx = union_find(len(pred_mentions), pred_links)

    clusters = []
    for group in clusters_idx:
        cluster = [pred_mentions[i] for i in group]
        if len(cluster) >= 2:
            clusters.append(cluster)

    return clusters


def clusters_to_token_map(clusters):
    """
    Converte clusters previstos em tags Entity= para CoNLL-U.
    Também filtra menções sobrepostas para evitar erro no corefud-scorer.
    """
    candidates = []

    for cid, cluster in enumerate(clusters, start=1):
        if len(cluster) < 2:
            continue

        eid = f"e{cid}"

        for s, e in cluster:
            s = int(s)
            e = int(e)

            if s < 0 or e < s:
                continue

            candidates.append((eid, s, e))

    # manter menções mais longas primeiro, depois as mais à esquerda
    candidates = sorted(
        set(candidates),
        key=lambda x: (-(x[2] - x[1]), x[1], x[2], x[0])
    )

    accepted = []

    for eid, s, e in candidates:
        overlaps = any(not (e < ps or s > pe) for _, ps, pe in accepted)
        if not overlaps:
            accepted.append((eid, s, e))

    # remover clusters que ficaram com menos de 2 menções
    by_eid = defaultdict(list)
    for eid, s, e in accepted:
        by_eid[eid].append((s, e))

    final_mentions = []
    new_cid = 1

    for old_eid in sorted(by_eid.keys()):
        spans = sorted(set(by_eid[old_eid]))
        if len(spans) < 2:
            continue

        eid = f"e{new_cid}"
        new_cid += 1

        for s, e in spans:
            final_mentions.append((eid, s, e))

    opens = defaultdict(list)
    closes = defaultdict(list)
    singles = defaultdict(list)

    for eid, s, e in final_mentions:
        if s == e:
            singles[s].append(eid)
        else:
            opens[s].append(eid)
            closes[e].append(eid)

    token_map = {}

    token_positions = set()
    token_positions.update(opens.keys())
    token_positions.update(closes.keys())
    token_positions.update(singles.keys())

    for tok in token_positions:
        tags = []

        for eid in sorted(singles[tok]):
            tags.append(f"({eid})")

        for eid in sorted(opens[tok]):
            tags.append(f"({eid}")

        for eid in sorted(closes[tok], reverse=True):
            tags.append(f"{eid})")

        token_map[tok] = "|".join(tags)

    return token_map

def write_predictions(gold_conllu, output_path, all_clusters):
    """
    Usa o gold .conllu como esqueleto e troca apenas o Entity= do MISC
    pelas predições do modelo.

    Importante:
    - mantém tokens, sent_id e newdoc iguais ao gold;
    - remove Entity= antigo;
    - escreve Entity= previsto.
    """
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

        if line.startswith("#"):
            out.append(line)
            continue

        if line.strip() == "":
            out.append(line)
            continue

        fields = line.split("\t")

        if len(fields) < 10:
            out.append(line)
            continue

        # pula multi-word tokens e empty nodes, se existirem
        if "-" in fields[0] or "." in fields[0]:
            out.append(line)
            continue

        old_misc = fields[9]

        if old_misc == "_" or old_misc == "":
            misc_parts = []
        else:
            misc_parts = [
                p for p in old_misc.split("|")
                if not p.startswith("Entity=")
            ]

        pred_tag = token_map.get(tok_in_doc, "")

        if pred_tag:
            misc_parts.append(f"Entity={pred_tag}")

        fields[9] = "|".join(misc_parts) if misc_parts else "_"

        out.append("\t".join(fields))
        tok_in_doc += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[OK] Predictions written to: {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--gold_conllu", required=True)
    parser.add_argument("--output_conllu", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--mention_thresh", type=float, default=0.0)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA:", torch.cuda.is_available())
    print("Device:", device)

    save_dir = Path(args.save_dir)

    if args.resume_from is None:
        best_path = save_dir / "best_epoch.txt"
        if not best_path.exists():
            raise FileNotFoundError(
                f"Could not find {best_path}. "
                "Pass --resume_from explicitly, e.g. checkpoint_epoch_1.pt."
            )

        best_epoch = int(best_path.read_text().strip())
        args.resume_from = str(save_dir / f"checkpoint_epoch_{best_epoch}.pt")

    print(f"[Checkpoint] {args.resume_from}")

    checkpoint = torch.load(args.resume_from, map_location=device)

    if "config" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'config'. Re-train saving config in checkpoint.")

    config = checkpoint["config"]

    print("[Config]")
    for key in sorted(config.keys()):
        print(f"  {key}: {config[key]}")

    # Resolve encoder para caminho local no cache, para evitar acesso à internet
    config["encoder_name"] = resolve_local_encoder_path(config["encoder_name"])

    tokenizer = AutoTokenizer.from_pretrained(
        config["encoder_name"],
        local_files_only=True,
    )
    model = CorefModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = CorefDataset(args.data_dir, args.split, config)
    print(f"[OK] Loaded {len(dataset)} docs from split '{args.split}'")

    all_clusters = []

    for i, ex in enumerate(dataset):
        clusters = predict_one_doc(
            model=model,
            ex=ex,
            tokenizer=tokenizer,
            config=config,
            device=device,
            mention_thresh=args.mention_thresh,
        )

        all_clusters.append(clusters)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"[{i+1}/{len(dataset)}] predicted clusters: {len(clusters)}", flush=True)

    write_predictions(
        gold_conllu=args.gold_conllu,
        output_path=args.output_conllu,
        all_clusters=all_clusters,
    )


if __name__ == "__main__":
    main()
