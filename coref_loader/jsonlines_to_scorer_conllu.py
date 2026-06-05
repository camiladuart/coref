import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def clean_doc_id(doc_id):
    return re.sub(r"[^A-Za-z0-9]", "", str(doc_id))


def filter_mentions_for_scorer(clusters, doc_id):
    """
    Recebe clusters do jsonlines:
      clusters = [[(s,e), (s,e)], ...]

    Devolve clusters sem sobreposição global, porque o udapi/corefud-scorer
    quebra com menções sobrepostas/nested em alguns casos.
    Mantém spans reais; não transforma tudo em singleton.
    """
    candidates = []

    for cid, cluster in enumerate(clusters, start=1):
        if len(cluster) < 2:
            continue

        eid = f"{clean_doc_id(doc_id)}e{cid}"

        for mention in cluster:
            if len(mention) != 2:
                continue

            s, e = int(mention[0]), int(mention[1])

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

    # remover cadeias que ficaram com menos de 2 menções depois do filtro
    by_eid = defaultdict(list)
    for eid, s, e in accepted:
        by_eid[eid].append((s, e))

    final = []
    for eid, spans in by_eid.items():
        spans = sorted(set(spans))
        if len(spans) >= 2:
            for s, e in spans:
                final.append((eid, s, e))

    return sorted(final, key=lambda x: (x[1], x[2], x[0]))


def build_misc(n_tokens, mentions):
    starts = defaultdict(list)
    ends = defaultdict(list)
    singles = defaultdict(list)

    for eid, s, e in mentions:
        if s < 0 or e >= n_tokens or e < s:
            continue

        if s == e:
            singles[s].append(eid)
        else:
            starts[s].append(eid)
            ends[e].append(eid)

    misc = ["_"] * n_tokens

    for i in range(n_tokens):
        items = []

        for eid in sorted(singles[i]):
            items.append(f"({eid})")

        for eid in sorted(starts[i]):
            items.append(f"({eid}")

        for eid in sorted(ends[i], reverse=True):
            items.append(f"{eid})")

        if items:
            misc[i] = "Entity=" + "|".join(items)

    return misc


def convert_file(input_path, output_path):
    docs_out = []

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            doc = json.loads(line)
            doc_id = doc.get("doc_key") or doc.get("doc_id") or doc.get("id")
            if doc_id is None:
                raise ValueError(f"Document without doc_key/doc_id/id in {input_path}")

            sentences = doc["sentences"]
            tokens = [tok for sent in sentences for tok in sent]
            clusters = doc.get("clusters", [])

            mentions = filter_mentions_for_scorer(clusters, doc_id)
            misc = build_misc(len(tokens), mentions)

            lines = []
            lines.append(f"# newdoc id = {doc_id}")
            lines.append(f"# sent_id = {doc_id}-0")
            lines.append("# text = " + " ".join(tokens))

            for i, tok in enumerate(tokens, start=1):
                if i == 1:
                    head = "0"
                    deprel = "root"
                else:
                    head = "1"
                    deprel = "dep"

                cols = [
                    str(i),
                    tok,
                    "_",
                    "_",
                    "_",
                    "_",
                    head,
                    deprel,
                    "_",
                    misc[i - 1],
                ]

                lines.append("\t".join(cols))

            docs_out.append("\n".join(lines))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("# global.Entity = eid\n")
        out.write("\n\n".join(docs_out))
        out.write("\n")

    print(f"[OK] {output_path}")
    print(f"     documents: {len(docs_out)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    for split in args.splits:
        inp = input_dir / f"{split}.jsonlines"
        out = output_dir / f"{split}.conllu"

        if not inp.exists():
            print(f"[SKIP] {inp} not found")
            continue

        convert_file(inp, out)


if __name__ == "__main__":
    main()
