"""
Converte train/dev/test.jsonlines para CoNLL-U com anotações Entity= no campo MISC.
Uso:
  python -m coref_loader.jsonlines_to_conllu \
    --input_dir /projects/.../coref_data/coref_pt \
    --output_dir /projects/.../coref_data/coref_pt_conllu \
    --splits train dev test
"""
import argparse, json
from pathlib import Path
from collections import defaultdict

def clusters_to_token_map(clusters, n_tokens):
    """
    clusters: lista de listas de (start, end) — offsets flat de token
    Devolve: dict token_idx -> lista de strings CoNLL-U Entity=
    ex: token 3 começa cluster 1 e termina cluster 2 -> ["(e1", "e2)"]
    """
    opens  = defaultdict(list)   # token_idx -> [chain_ids que abrem aqui]
    closes = defaultdict(list)   # token_idx -> [chain_ids que fecham aqui]
    singles = defaultdict(list)  # token_idx -> [chain_ids singleton aqui]

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
        for cid in singles[tok]:
            tags.append(f"(e{cid})")
        for cid in opens[tok]:
            tags.append(f"(e{cid}")
        for cid in closes[tok]:
            tags.append(f"e{cid})")
        token_map[tok] = "|".join(tags)
    return token_map

def jsonlines_to_conllu(input_path, output_path):
    output_lines = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_key = doc["doc_key"]
            sentences = doc["sentences"]
            clusters = doc.get("clusters", [])

            # offsets flat
            n_tokens = sum(len(s) for s in sentences)
            token_map = clusters_to_token_map(clusters, n_tokens)

            output_lines.append(f"# newdoc id = {doc_key}")
            tok_offset = 0
            sent_idx = 0
            for sent in sentences:
                output_lines.append(f"# sent_id = {doc_key}-{sent_idx}")
                output_lines.append(f"# text = {' '.join(sent)}")
                for i, tok in enumerate(sent):
                    flat_idx = tok_offset + i
                    coref_tag = token_map.get(flat_idx, "")
                    misc = f"Entity={coref_tag}" if coref_tag else "_"
                    # CoNLL-U: ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC
                    output_lines.append(
                        f"{i+1}\t{tok}\t_\t_\t_\t_\t_\t_\t_\t{misc}"
                    )
                output_lines.append("")  # linha vazia entre frases
                tok_offset += len(sent)
                sent_idx += 1

    Path(output_path).write_text("\n".join(output_lines), encoding="utf-8")
    print(f"[OK] {output_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        inp = Path(args.input_dir) / f"{split}.jsonlines"
        if not inp.exists():
            print(f"[SKIP] {inp} não existe")
            continue
        jsonlines_to_conllu(inp, out / f"{split}.conllu")

if __name__ == "__main__":
    main()
