import os
import json
import argparse
from collections import defaultdict


def parse_coref_column(coref_str, token_index, open_mentions):
    if coref_str == "-" or coref_str == "_":
        return

    parts = coref_str.split("|")

    for part in parts:
        if part.startswith("(") and part.endswith(")"):
            cluster_id = part[1:-1]
            open_mentions[cluster_id].append((token_index, token_index))

        elif part.startswith("("):
            cluster_id = part[1:]
            open_mentions[cluster_id].append((token_index, None))

        elif part.endswith(")"):
            cluster_id = part[:-1]
            for i in range(len(open_mentions[cluster_id]) - 1, -1, -1):
                start, end = open_mentions[cluster_id][i]
                if end is None:
                    open_mentions[cluster_id][i] = (start, token_index)
                    break


def convert_semeval_to_jsonlines(input_path, output_path):
    documents = []
    current_tokens = []
    current_clusters = defaultdict(list)
    open_mentions = defaultdict(list)
    token_index = 0

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#begin document"):
                if current_tokens:
                    clusters = []
                    for cluster_id, mentions in open_mentions.items():
                        spans = []
                        for start, end in mentions:
                            if end is not None:
                                spans.append([start, end])
                        if spans:
                            clusters.append(spans)

                    documents.append({
                        "doc_key": f"doc_{len(documents)}",
                        "tokens": current_tokens,
                        "clusters": clusters
                    })

                current_tokens = []
                current_clusters = defaultdict(list)
                open_mentions = defaultdict(list)
                token_index = 0
                continue

            if line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            token = parts[1]
            coref = parts[-1]

            current_tokens.append(token)
            parse_coref_column(coref, token_index, open_mentions)
            token_index += 1

    # salvar último documento
    if current_tokens:
        clusters = []
        for cluster_id, mentions in open_mentions.items():
            spans = []
            for start, end in mentions:
                if end is not None:
                    spans.append([start, end])
            if spans:
                clusters.append(spans)

        documents.append({
            "doc_key": f"doc_{len(documents)}",
            "sentences": [current_tokens],
            "clusters": clusters
        })

    with open(output_path, "w", encoding="utf-8") as out:
        for doc in documents:
            out.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"{len(documents)} docs salvos em {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    input_file = os.path.join(args.input_dir, "Corref-PT-SemEval.txt")
    output_file = os.path.join(args.output_dir, f"{args.split_name}.jsonlines")

    convert_semeval_to_jsonlines(input_file, output_file)
